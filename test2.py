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

# Mahalanobis distance check
def mahalanobis_distance(x, P_inv, y):
    diff = np.array(x) - np.array(y)
    return np.sqrt(np.dot(np.dot(diff.T, P_inv), diff))

# Cost matrix for Munkres algorithm (position + Doppler)
def create_cost_matrix(track_positions, report_positions, track_dopplers, report_dopplers, doppler_weight=1.0):
    num_tracks = len(track_positions)
    num_reports = len(report_positions)
    cost_matrix = np.zeros((num_tracks, num_reports))

    for i, (track_pos, track_doppler) in enumerate(zip(track_positions, track_dopplers)):
        for j, (report_pos, report_doppler) in enumerate(zip(report_positions, report_dopplers)):
            spatial_distance = np.linalg.norm(np.array(track_pos) - np.array(report_pos))
            doppler_distance = doppler_weight * abs(track_doppler - report_doppler)
            cost_matrix[i, j] = spatial_distance + doppler_distance

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
    return next_id

# Helper to release (remove) a track ID
def release_track_id(track_id_list, track_id):
    if track_id in track_id_list:
        track_id_list.remove(track_id)

# Track initialization with Munkres data association and state transitions
def initialize_tracks_with_munkres(measurements, doppler_threshold, range_threshold, firm_threshold, time_threshold):
    tracks = []
    track_id_list = []
    miss_counts = {}
    hit_counts = {}
    tentative_ids = {}
    firm_ids = set()
    state_map = {}

    # State progression (dynamic)
    state_progression = {
        3: ['Poss1', 'Tentative1', 'Firm'],
        5: ['Poss1', 'Poss2', 'Tentative1', 'Firm'],
        7: ['Poss1', 'Poss2', 'Tentative1', 'Tentative2', 'Firm']
    }
    progression_states = state_progression[firm_threshold]

    # Get threshold for track expiry (misses)
    def get_miss_threshold(current_state, firm_threshold):
        if current_state is None:
            return firm_threshold
        if current_state.startswith('Poss'):
            return 1
        elif current_state.startswith('Tentative'):
            return 2 if firm_threshold == 3 else 3
        elif current_state == 'Firm':
            return 3 if firm_threshold == 3 else 5

    for i, measurement in enumerate(measurements):
        measurement_cartesian = sph2cart(measurement[0], measurement[1], measurement[2])
        measurement_doppler = measurement[3]
        measurement_time = measurement[4]

        # Extract positions of existing tracks for Munkres cost matrix
        track_positions, track_dopplers, track_times = [], [], []
        for track in tracks:
            if track:
                last_measurement = track['measurements'][-1][0]  # Most recent measurement
                last_cartesian = sph2cart(last_measurement[0], last_measurement[1], last_measurement[2])
                track_positions.append(last_cartesian)
                track_dopplers.append(last_measurement[3])
                track_times.append(last_measurement[4])

        # Perform Munkres data association
        assigned_tracks = set()
        if track_positions:
            # Filter out expired tracks based on time threshold
            active_tracks = [i for i, t in enumerate(track_times) if measurement_time - t <= time_threshold]
            track_positions = [track_positions[i] for i in active_tracks]
            track_dopplers = [track_dopplers[i] for i in active_tracks]

            # Create cost matrix
            cost_matrix = create_cost_matrix(track_positions, [measurement_cartesian], track_dopplers, [measurement_doppler])
            assignments = munkres_algorithm(cost_matrix)

            for track_idx, meas_idx in assignments:
                distance = cost_matrix[track_idx, meas_idx]
                if distance < range_threshold:
                    track = tracks[track_idx]
                    track['measurements'].append((measurement, state_map[track['track_id']]))
                    assigned_tracks.add(track_idx)

                    # Update hit/miss counts
                    track_id = track['track_id']
                    if track_id in tentative_ids:
                        hit_counts[track_id] += 1
                        miss_counts[track_id] = 0

                        # Update track state based on hit counts
                        if hit_counts[track_id] >= firm_threshold:
                            firm_ids.add(track_id)
                            state_map[track_id] = 'Firm'
                        else:
                            state_map[track_id] = progression_states[hit_counts[track_id] - 1]

        # Initialize a new track if no association
        if len(assigned_tracks) == 0:
            new_track_id = get_next_track_id(track_id_list)
            tracks.append({
                'track_id': new_track_id,
                'measurements': [(measurement, progression_states[0])]
            })
            miss_counts[new_track_id] = 0
            hit_counts[new_track_id] = 1
            tentative_ids[new_track_id] = True
            state_map[new_track_id] = progression_states[0]

        # Handle missed detections for existing tracks
        for track in tracks:
            track_id = track['track_id']
            if track_id not in assigned_tracks:
                current_state = state_map.get(track_id)
                miss_threshold = get_miss_threshold(current_state, firm_threshold)
                miss_counts[track_id] += 1
                if miss_counts[track_id] >= miss_threshold:
                    tracks[tracks.index(track)] = None
                    release_track_id(track_id_list, track_id)
                    state_map.pop(track_id, None)

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

# Main function
def main():
    # File path for measurements CSV
    file_path = 'ttk.csv'

    # Read measurements from CSV
    measurements = read_measurements_from_csv(file_path)

    # Initialize parameters
    doppler_threshold = 5.0
    range_threshold = 10.0
    firm_threshold = 5
    time_threshold = 0.050  # 50 ms

    # Initialize tracks with Munkres data association
    tracks, track_id_list, miss_counts, hit_counts, firm_ids, state_map, progression_states = initialize_tracks_with_munkres(
        measurements, doppler_threshold, range_threshold, firm_threshold, time_threshold
    )

    # Output track states for visualization or further processing
    for track in tracks:
        if track:
            track_id = track['track_id']
            track_state = state_map[track_id]
            print(f"Track {track_id} is in state {track_state}")
            print("Measurements (range, azimuth, elevation, doppler):")
            for measurement, state in track['measurements']:
                print(f"Measurement: {measurement}, State: {state}")

if __name__ == "__main__":
    main()
