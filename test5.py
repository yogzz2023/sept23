import numpy as np
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

# Kalman Filter class
class KalmanFilter:
    def __init__(self, initial_state, initial_covariance):
        self.x = initial_state
        self.P = initial_covariance
        self.F = np.eye(len(initial_state))  # State transition matrix
        self.H = np.eye(len(initial_state))  # Measurement function
        self.R = np.eye(len(initial_state))  # Measurement uncertainty
        self.Q = np.eye(len(initial_state))  # Process noise covariance

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = np.dot(np.eye(len(self.x)) - np.dot(K, self.H), self.P)

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

# Track initialization with Munkres data association
def initialize_tracks_with_munkres(measurements, doppler_threshold, range_threshold, firm_threshold, time_threshold):
    tracks = []
    track_id_list = []
    miss_counts = {}
    hit_counts = {}
    firm_ids = set()
    state_map = {}

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
        measurement_cartesian = sph2cart(measurement[1], measurement[2], measurement[0])
        measurement_doppler = measurement[3]
        measurement_time = measurement[4]

        # Extract positions of existing tracks for Munkres cost matrix
        track_positions = []
        for track in tracks:
            if track:
                last_measurement = track['measurements'][-1][0]  # Most recent measurement
                last_cartesian = sph2cart(last_measurement[0], last_measurement[1], last_measurement[2])
                track_positions.append(last_cartesian)

        # Debug: Check number of track positions
        print(f"Number of existing tracks: {len(track_positions)}")

        # Perform Munkres data association if there are existing tracks
        assigned_tracks = set()
        if track_positions:
            # Create the cost matrix between current measurement and existing track positions
            cost_matrix = create_cost_matrix(track_positions, [measurement_cartesian])
            print(f"Cost matrix: {cost_matrix}")  # Debug output
            assignments = munkres_algorithm(cost_matrix)

            for track_idx, meas_idx in assignments:
                distance = cost_matrix[track_idx, meas_idx]
                print(f"Track {track_idx} assigned to measurement {meas_idx} with distance {distance}")  # Debug output
                if distance < range_threshold:  # Apply gating (range threshold)
                    track = tracks[track_idx]

                    # Doppler correlation check
                    last_measurement = track['measurements'][-1][0]
                    last_doppler = last_measurement[3]
                    doppler_correlated = doppler_correlation(measurement_doppler, last_doppler, doppler_threshold)

                    # If doppler matches and gating is satisfied, update the track
                    if doppler_correlated:
                        track['measurements'].append((measurement, state_map[track['track_id']]))
                        assigned_tracks.add(track_idx)

                        # Update hit/miss counts
                        if track['track_id'] in hit_counts:
                            hit_counts[track['track_id']] += 1
                            miss_counts[track['track_id']] = 0
                            # State progression based on hit count
                            if hit_counts[track['track_id']] >= firm_threshold:
                                firm_ids.add(track['track_id'])
                                state_map[track['track_id']] = 'Firm'
                            else:
                                state_map[track['track_id']] = progression_states[hit_counts[track['track_id']] - 1]

        # If measurement wasn't assigned to any track, initialize a new one
        if len(assigned_tracks) == 0:
            new_track_id, new_track_idx = get_next_track_id(track_id_list)
            tracks.append({
                'track_id': new_track_id,
                'measurements': [(measurement, progression_states[0])]
            })
            miss_counts[new_track_id] = 0
            hit_counts[new_track_id] = 1
            state_map[new_track_id] = progression_states[0]

        # Handle missed detections (not assigned in Munkres)
        for track_idx, track in enumerate(tracks):
            if track_idx not in assigned_tracks and track:
                current_state = state_map.get(track['track_id'])
                miss_threshold = get_miss_threshold(current_state, firm_threshold)
                miss_counts[track['track_id']] += 1
                if miss_counts[track['track_id']] >= miss_threshold:
                    print(f"Track {track['track_id']} is removed due to missed threshold")  # Debug output
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
            doppler = float(row[11])  # Doppler
            time = float(row[10])  # Measurement time
            measurements.append((r, az, el, doppler, time))
    return measurements

# Main function
def main():
    # File path for measurements CSV
    file_path = 'ttk.csv'

    # Read measurements from CSV
    measurements = read_measurements_from_csv(file_path)

    # Initialize parameters
    doppler_threshold = 5.0
    range_threshold = 10.0
    firm_threshold = 3
    time_threshold = 0.05  # 50 milliseconds

    # Perform track initiation and filtering
    tracks, track_id_list, miss_counts, hit_counts, firm_ids, state_map, progression_states = initialize_tracks_with_munkres(
        measurements, doppler_threshold, range_threshold, firm_threshold, time_threshold)

    # Print track details
    print("\nTrack details after initiation:")
    for track in tracks:
        if track:
            print(f"Track ID: {track['track_id']}, State: {state_map[track['track_id']]}, Measurements: {track['measurements']}, Hit count: {hit_counts[track['track_id']]}, Miss count: {miss_counts[track['track_id']]}")

if __name__ == '__main__':
    main()
