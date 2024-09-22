# Import necessary libraries
import numpy as np
from munkres import Munkres

# Utility functions for spherical to cartesian conversion
def sph2cart(range, azimuth, elevation):
    x = range * np.cos(elevation) * np.cos(azimuth)
    y = range * np.cos(elevation) * np.sin(azimuth)
    z = range * np.sin(elevation)
    return np.array([x, y, z])

# Create a cost matrix for Munkres algorithm (track measurement association)
def create_cost_matrix(track_positions, measurement_positions, track_dopplers, measurement_dopplers):
    cost_matrix = np.zeros((len(track_positions), len(measurement_positions)))

    for i, track_pos in enumerate(track_positions):
        for j, meas_pos in enumerate(measurement_positions):
            pos_diff = np.linalg.norm(track_pos - meas_pos)
            doppler_diff = abs(track_dopplers[i] - measurement_dopplers[j])
            cost_matrix[i, j] = pos_diff + doppler_diff  # Simple cost combining position and doppler differences

    return cost_matrix

# Munkres algorithm for optimal assignment
def munkres_algorithm(cost_matrix):
    m = Munkres()
    return m.compute(cost_matrix.tolist())

# Get next available track ID
def get_next_track_id(track_id_list):
    new_track_id = len(track_id_list)
    track_id_list.append(new_track_id)
    return new_track_id

# Release track ID
def release_track_id(track_id_list, track_id):
    if track_id in track_id_list:
        track_id_list.remove(track_id)

# Initialize tracks with Munkres data association and state transitions
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
            if track:  # Check if the track is not None
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
        for idx, track in enumerate(tracks):
            if track and track['track_id'] not in assigned_tracks:
                track_id = track['track_id']
                current_state = state_map.get(track_id)
                miss_threshold = get_miss_threshold(current_state, firm_threshold)
                miss_counts[track_id] += 1
                if miss_counts[track_id] >= miss_threshold:
                    tracks[idx] = None  # Mark track for deletion
                    release_track_id(track_id_list, track_id)
                    state_map.pop(track_id, None)

    # Remove None entries from tracks list
    tracks = [track for track in tracks if track is not None]

    return tracks, track_id_list, miss_counts, hit_counts, firm_ids, state_map, progression_states

# Main function to demonstrate the initialization process
def main():
    # Example measurements: (range, azimuth, elevation, doppler, time)
    measurements = [
        (10.0, 0.1, 0.05, 1.2, 0),
        (10.5, 0.12, 0.06, 1.1, 1),
        (11.0, 0.13, 0.07, 1.3, 2),
        # Add more measurements as needed
    ]

    doppler_threshold = 1.5
    range_threshold = 1.0
    firm_threshold = 5  # Progression stages depend on this value
    time_threshold = 5  # Time threshold for track expiry

    tracks, track_id_list, miss_counts, hit_counts, firm_ids, state_map, progression_states = initialize_tracks_with_munkres(
        measurements, doppler_threshold, range_threshold, firm_threshold, time_threshold)

    # Display results
    print("Tracks:", tracks)
    print("Track ID List:", track_id_list)
    print("Miss Counts:", miss_counts)
    print("Hit Counts:", hit_counts)
    print("Firm IDs:", firm_ids)
    print("State Map:", state_map)
    print("Progression States:", progression_states)

if __name__ == "__main__":
    main()
