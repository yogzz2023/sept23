import numpy as np
import math
import csv
from scipy.optimize import linear_sum_assignment

# Spherical to Cartesian conversion
def sph2cart(az, el, r):
    x = r * np.cos(el * np.pi / 180) * np.sin(az * np.pi / 180)
    y = r * np.cos(el * np.pi / 180) * np.cos(az * np.pi / 180)
    z = r * np.sin(el * np.pi / 180)
    return x, y, z

# Doppler correlation check
def doppler_correlation(doppler1, doppler2, threshold):
    return abs(doppler1 - doppler2) < threshold

# Kalman filter initialization
def initialize_kalman():
    # State vector [x, y, z, vx, vy, vz] for 3D position and velocity
    x = np.zeros((6, 1))  # Initial state (can be set based on the first measurement)
    P = np.eye(6) * 1000  # Initial uncertainty
    F = np.eye(6)  # State transition matrix (adjust with velocity)
    Q = np.eye(6) * 0.1  # Process noise covariance
    H = np.zeros((3, 6))  # Measurement matrix
    H[:3, :3] = np.eye(3)  # Assuming we only measure position
    R = np.eye(3) * 5  # Measurement noise covariance
    return x, P, F, Q, H, R

# Kalman filter predict step
def predict_kalman(x, P, F, Q):
    x = F @ x  # Predict the next state
    P = F @ P @ F.T + Q  # Predict the uncertainty
    return x, P

# Kalman filter update step
def update_kalman(x, P, z, H, R):
    y = z.reshape(-1, 1) - H @ x  # Measurement residual
    S = H @ P @ H.T + R  # Residual covariance
    K = P @ H.T @ np.linalg.inv(S)  # Kalman gain
    x = x + K @ y  # Updated state estimate
    P = (np.eye(len(x)) - K @ H) @ P  # Updated uncertainty
    return x, P

# Cost matrix for Munkres algorithm
def create_cost_matrix(track_positions, report_positions):
    num_tracks = len(track_positions)
    num_reports = len(report_positions)
    cost_matrix = np.zeros((num_tracks, num_reports))

    for i, track_pos in enumerate(track_positions):
        for j, report_pos in enumerate(report_positions):
            cost_matrix[i, j] = np.linalg.norm(np.array(track_pos) - np.array(report_pos))

    return cost_matrix

# Munkres algorithm (Hungarian)
def munkres_algorithm(cost_matrix):
    track_indices, report_indices = linear_sum_assignment(cost_matrix)
    return list(zip(track_indices, report_indices))

# Helper to get next track ID
def get_next_track_id(track_id_list):
    next_id = 0
    while next_id in track_id_list:
        next_id += 1
    track_id_list.append(next_id)
    return next_id, track_id_list.index(next_id)

# Helper to release (remove) a track ID
def release_track_id(track_id_list, idx):
    if idx < len(track_id_list):
        track_id_list.pop(idx)

# Track initialization with Munkres data association and Kalman filter
def initialize_tracks_with_munkres_and_kalman(measurements, doppler_threshold, range_threshold, firm_threshold, time_threshold):
    tracks = []
    track_id_list = []
    miss_counts = {}
    hit_counts = {}
    tentative_ids = {}
    firm_ids = set()
    state_map = {}
    kalman_data = {}  # Store Kalman filter states

    state_progression = {
        3: ['Poss1', 'Tentative1', 'Firm'],
        5: ['Poss1', 'Poss2', 'Tentative1', 'Tentative2', 'Firm'],
        7: ['Poss1', 'Poss2', 'Tentative1', 'Tentative2', 'Tentative3', 'Firm']
    }
    progression_states = state_progression[firm_threshold]

    def get_miss_threshold(current_state, firm_threshold):
        if current_state is None:
            return firm_threshold
        if current_state.startswith('Poss'):
            return 1
        elif current_state.startswith('Tentative'):
            return 2 if firm_threshold == 3 else 3
        elif current_state == 'Firm':
            return 3 if firm_threshold == 3 else 5

    # Debug: Check the number of measurements
    print(f"Number of measurements: {len(measurements)}")

    for i, measurement in enumerate(measurements):
        print(f"Processing measurement {i+1}: {measurement}")  # Debug output
        measurement_cartesian = sph2cart(measurement[0], measurement[1], measurement[2])
        measurement_doppler = measurement[3]
        measurement_time = measurement[4]

        # Extract positions of existing tracks for Munkres cost matrix
        track_positions = []
        for track in tracks:
            if track:
                last_measurement = track['measurements'][-1][0]  # Most recent measurement
                last_cartesian = sph2cart(last_measurement[0], last_measurement[1], last_measurement[2])
                track_positions.append(last_cartesian)

        # Perform Munkres data association if there are existing tracks
        assigned_tracks = set()
        if track_positions:
            cost_matrix = create_cost_matrix(track_positions, [measurement_cartesian])
            assignments = munkres_algorithm(cost_matrix)

            for track_idx, meas_idx in assignments:
                distance = cost_matrix[track_idx, meas_idx]
                if distance < range_threshold:
                    track = tracks[track_idx]
                    last_measurement = track['measurements'][-1][0]
                    last_doppler = last_measurement[3]
                    doppler_correlated = doppler_correlation(measurement_doppler, last_doppler, doppler_threshold)

                    if doppler_correlated:
                        track['measurements'].append((measurement, state_map[track['track_id']]))
                        assigned_tracks.add(track_idx)

                        if len(track['measurements']) >= 3:
                            # Kalman filtering: predict and update when track has 3 or more measurements
                            if track['track_id'] not in kalman_data:
                                x, P, F, Q, H, R = initialize_kalman()
                                kalman_data[track['track_id']] = (x, P, F, Q, H, R)

                            # Kalman prediction and update
                            x, P, F, Q, H, R = kalman_data[track['track_id']]
                            x, P = predict_kalman(x, P, F, Q)
                            x, P = update_kalman(x, P, np.array(measurement_cartesian), H, R)
                            kalman_data[track['track_id']] = (x, P, F, Q, H, R)

                            # Update track with Kalman filtered position
                            track['kalman_filtered_position'] = x[:3].flatten()

        # If measurement wasn't assigned to any track, initialize a new one
        if len(assigned_tracks) == 0:
            new_track_id, new_track_idx = get_next_track_id(track_id_list)
            tracks.append({
                'track_id': new_track_id,
                'measurements': [(measurement, progression_states[0])]
            })
            miss_counts[new_track_id] = 0
            hit_counts[new_track_id] = 1
            tentative_ids[new_track_id] = True
            state_map[new_track_id] = progression_states[0]

        # Handle missed detections (not assigned in Munkres)
        for track_idx, track in enumerate(tracks):
            if track_idx not in assigned_tracks and track:
                current_state = state_map.get(track['track_id'])
                miss_threshold = get_miss_threshold(current_state, firm_threshold)
                miss_counts[track['track_id']] += 1
                if miss_counts[track['track_id']] >= miss_threshold:
                    tracks[track_idx] = None
                    release_track_id(track_id_list, track_idx)
                    state_map.pop(track['track_id'], None)

    return tracks, track_id_list, miss_counts, hit_counts, firm_ids, state_map, progression_states

# Read measurements from a CSV file
def read_measurements_from_csv(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            r = float(row[7])  # Range
            az = float(row[8])  # Azimuth
            el = float(row[9])  # Elevation
            doppler = float(row[10])  # Doppler
            time = float(row[11])  # Measurement time
            measurements.append((r, az, el, doppler, time))
    return measurements

def main():
    file_path = 'ttk.csv'  # Update with your CSV file path
    measurements = read_measurements_from_csv(file_path)

    # Define thresholds
    doppler_threshold = 10.0  # Adjust based on your requirements
    range_threshold = 50.0  # Adjust based on your requirements
    firm_threshold = 3  # Number of measurements to transition to firm state

    # Initialize tracks with the measurements
    tracks, track_id_list, miss_counts, hit_counts, firm_ids, state_map, progression_states = (
        initialize_tracks_with_munkres_and_kalman(measurements, doppler_threshold, range_threshold, firm_threshold, firm_threshold)
    )

    # Output the results
    print("Tracks after processing:")
    for track in tracks:
        if track is not None:
            print(f"Track ID: {track['track_id']}, Measurements: {track['measurements']}, Kalman Filtered Position: {track.get('kalman_filtered_position', 'N/A')}")

    print("\nMiss Counts:", miss_counts)
    print("Hit Counts:", hit_counts)
    print("Firm IDs:", firm_ids)

if __name__ == "__main__":
    main()
