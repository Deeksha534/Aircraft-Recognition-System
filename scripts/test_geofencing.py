"""
test_geofencing.py
Comprehensive unit test suite for GeofenceManager and point-in-polygon logic.
Validates all 12 Phase 4 test scenarios using synthetic coordinates.
"""

import sys
import os
import unittest
import numpy as np

# Ensure scripts directory is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from geofencing import (
    GeofenceManager,
    GeofenceZone,
    GeofenceAlert,
    point_in_polygon,
    validate_polygon_coords,
)


class TestGeofencing(unittest.TestCase):

    def setUp(self):
        """Set up standard test zones."""
        # Rectangle from (100, 100) to (300, 300)
        self.zone1 = GeofenceZone(
            id="zone_alpha",
            name="Alpha Sector",
            polygon=[(100.0, 100.0), (300.0, 100.0), (300.0, 300.0), (100.0, 300.0)],
            enabled=True,
        )
        # Second non-overlapping zone from (500, 500) to (700, 700)
        self.zone2 = GeofenceZone(
            id="zone_bravo",
            name="Bravo Sector",
            polygon=[(500.0, 500.0), (700.0, 500.0), (700.0, 700.0), (500.0, 700.0)],
            enabled=True,
        )

    def test_01_point_outside_produces_no_alert(self):
        """Scenario 1: A point outside a polygon produces no entry alert."""
        gm = GeofenceManager(zones=[self.zone1])
        # Track 1 is outside at (50, 50)
        alerts = gm.update_track(track_id=1, center_point=(50, 50), label="F-16", frame_idx=1)
        self.assertEqual(len(alerts), 0)
        self.assertEqual(len(gm.alerts_history), 0)
        # Second step still outside at (60, 60)
        alerts2 = gm.update_track(track_id=1, center_point=(60, 60), label="F-16", frame_idx=2)
        self.assertEqual(len(alerts2), 0)
        self.assertEqual(len(gm.alerts_history), 0)

    def test_02_outside_to_inside_produces_exactly_one_alert(self):
        """Scenario 2: An observed outside-to-inside transition produces exactly one alert."""
        gm = GeofenceManager(zones=[self.zone1])
        # Start outside
        gm.update_track(track_id=1, center_point=(50, 50), label="F-16", frame_idx=1)
        # Move inside (200, 200)
        alerts = gm.update_track(track_id=1, center_point=(200, 200), label="F-16", frame_idx=2)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(len(gm.alerts_history), 1)
        alert = alerts[0]
        self.assertEqual(alert.track_id, 1)
        self.assertEqual(alert.label, "F-16")
        self.assertEqual(alert.zone_id, "zone_alpha")
        self.assertEqual(alert.zone_name, "Alpha Sector")
        self.assertEqual(alert.event_type, "ZONE_ENTRY")
        self.assertEqual(alert.frame_idx, 2)
        self.assertEqual(alert.position, (200, 200))

    def test_03_remaining_inside_produces_no_repeated_alert(self):
        """Scenario 3: Remaining inside produces no repeated alert."""
        gm = GeofenceManager(zones=[self.zone1])
        # Step 1: Outside
        gm.update_track(track_id=1, center_point=(50, 50), label="F-16", frame_idx=1)
        # Step 2: Enters inside -> exactly 1 alert
        alerts2 = gm.update_track(track_id=1, center_point=(200, 200), label="F-16", frame_idx=2)
        self.assertEqual(len(alerts2), 1)
        # Step 3: Still inside at (210, 210) -> 0 alerts
        alerts3 = gm.update_track(track_id=1, center_point=(210, 210), label="F-16", frame_idx=3)
        self.assertEqual(len(alerts3), 0)
        # Step 4: Still inside at (220, 220) -> 0 alerts
        alerts4 = gm.update_track(track_id=1, center_point=(220, 220), label="F-16", frame_idx=4)
        self.assertEqual(len(alerts4), 0)
        # Total historical alerts must remain 1
        self.assertEqual(len(gm.alerts_history), 1)

    def test_04_leaving_zone_produces_no_alert(self):
        """Scenario 4: Leaving a zone produces no entry alert."""
        gm = GeofenceManager(zones=[self.zone1])
        # Step 1: Outside
        gm.update_track(track_id=1, center_point=(50, 50), label="B-2", frame_idx=1)
        # Step 2: Inside -> 1 alert
        gm.update_track(track_id=1, center_point=(200, 200), label="B-2", frame_idx=2)
        # Step 3: Moves outside to (50, 50) -> 0 alerts
        alerts_exit = gm.update_track(track_id=1, center_point=(50, 50), label="B-2", frame_idx=3)
        self.assertEqual(len(alerts_exit), 0)
        self.assertEqual(len(gm.alerts_history), 1)
        self.assertFalse(gm.is_track_inside(1, "zone_alpha"))

    def test_05_reentering_after_exiting_produces_new_alert(self):
        """Scenario 5: Re-entering after exiting produces another alert."""
        gm = GeofenceManager(zones=[self.zone1])
        # Start outside -> Enter (Alert 1)
        gm.update_track(track_id=1, center_point=(50, 50), label="Su-57", frame_idx=1)
        alerts1 = gm.update_track(track_id=1, center_point=(200, 200), label="Su-57", frame_idx=2)
        self.assertEqual(len(alerts1), 1)
        # Leave zone
        alerts_exit = gm.update_track(track_id=1, center_point=(50, 50), label="Su-57", frame_idx=3)
        self.assertEqual(len(alerts_exit), 0)
        # Re-enter zone -> Alert 2
        alerts2 = gm.update_track(track_id=1, center_point=(250, 250), label="Su-57", frame_idx=4)
        self.assertEqual(len(alerts2), 1)
        self.assertEqual(len(gm.alerts_history), 2)
        self.assertEqual(alerts2[0].frame_idx, 4)

    def test_06_two_aircraft_tracked_independently(self):
        """Scenario 6: Two aircraft are tracked independently."""
        gm = GeofenceManager(zones=[self.zone1])
        # Both start outside
        gm.update_track(track_id=1, center_point=(50, 50), label="F-22", frame_idx=1)
        gm.update_track(track_id=2, center_point=(60, 60), label="A-10", frame_idx=1)
        # Track 1 enters zone
        alerts_t1 = gm.update_track(track_id=1, center_point=(200, 200), label="F-22", frame_idx=2)
        self.assertEqual(len(alerts_t1), 1)
        self.assertEqual(alerts_t1[0].track_id, 1)
        # Track 2 stays outside
        alerts_t2 = gm.update_track(track_id=2, center_point=(70, 70), label="A-10", frame_idx=2)
        self.assertEqual(len(alerts_t2), 0)
        # Next frame: Track 1 stays inside (no alert), Track 2 enters zone (alert for Track 2)
        alerts_t1_stay = gm.update_track(track_id=1, center_point=(210, 210), label="F-22", frame_idx=3)
        self.assertEqual(len(alerts_t1_stay), 0)
        alerts_t2_enter = gm.update_track(track_id=2, center_point=(150, 150), label="A-10", frame_idx=3)
        self.assertEqual(len(alerts_t2_enter), 1)
        self.assertEqual(alerts_t2_enter[0].track_id, 2)
        self.assertEqual(len(gm.alerts_history), 2)

    def test_07_multiple_zones_handled_independently(self):
        """Scenario 7: Multiple zones are handled independently."""
        gm = GeofenceManager(zones=[self.zone1, self.zone2])
        # Start outside both zones
        gm.update_track(track_id=1, center_point=(0, 0), label="Eurofighter", frame_idx=1)
        # Enters zone 1, outside zone 2
        alerts_z1 = gm.update_track(track_id=1, center_point=(200, 200), label="Eurofighter", frame_idx=2)
        self.assertEqual(len(alerts_z1), 1)
        self.assertEqual(alerts_z1[0].zone_id, "zone_alpha")
        # Leaves zone 1 and enters zone 2
        alerts_z2 = gm.update_track(track_id=1, center_point=(600, 600), label="Eurofighter", frame_idx=3)
        self.assertEqual(len(alerts_z2), 1)
        self.assertEqual(alerts_z2[0].zone_id, "zone_bravo")
        self.assertFalse(gm.is_track_inside(1, "zone_alpha"))
        self.assertTrue(gm.is_track_inside(1, "zone_bravo"))

    def test_08_disabled_zones_produce_no_alerts(self):
        """Scenario 8: Disabled zones produce no alerts."""
        disabled_zone = GeofenceZone(
            id="zone_disabled",
            name="Deactivated Sector",
            polygon=[(100.0, 100.0), (300.0, 100.0), (300.0, 300.0), (100.0, 300.0)],
            enabled=False,
        )
        gm = GeofenceManager(zones=[disabled_zone])
        # Start outside
        gm.update_track(track_id=1, center_point=(50, 50), label="F-35", frame_idx=1)
        # Move inside disabled zone
        alerts = gm.update_track(track_id=1, center_point=(200, 200), label="F-35", frame_idx=2)
        self.assertEqual(len(alerts), 0)
        self.assertEqual(len(gm.alerts_history), 0)

    def test_09_track_first_observed_inside_produces_no_alert(self):
        """Scenario 9: A track first observed inside does not generate a false entry alert."""
        gm = GeofenceManager(zones=[self.zone1])
        # Track 1 appears inside at (200, 200) without previous outside observation
        alerts_first = gm.update_track(track_id=1, center_point=(200, 200), label="Mirage", frame_idx=1)
        self.assertEqual(len(alerts_first), 0)
        self.assertEqual(len(gm.alerts_history), 0)
        self.assertTrue(gm.is_track_inside(1, "zone_alpha"))
        # Second frame still inside -> still no alert
        alerts_second = gm.update_track(track_id=1, center_point=(210, 210), label="Mirage", frame_idx=2)
        self.assertEqual(len(alerts_second), 0)
        # Leaves zone
        alerts_leave = gm.update_track(track_id=1, center_point=(50, 50), label="Mirage", frame_idx=3)
        self.assertEqual(len(alerts_leave), 0)
        # Now that an outside state has been observed, re-entering DOES trigger an alert!
        alerts_reenter = gm.update_track(track_id=1, center_point=(200, 200), label="Mirage", frame_idx=4)
        self.assertEqual(len(alerts_reenter), 1)

    def test_10_invalid_polygon_configuration_handled_safely(self):
        """Scenario 10: Invalid polygon configuration is handled safely."""
        # Less than 3 points
        valid, msg = validate_polygon_coords([(100, 100), (200, 200)])
        self.assertFalse(valid)
        # Non-numeric points
        valid2, msg2 = validate_polygon_coords([(100, 100), ("bad", 200), (300, 300)])
        self.assertFalse(valid2)
        # Zero area collinear points
        valid3, msg3 = validate_polygon_coords([(100, 100), (200, 100), (300, 100)])
        self.assertFalse(valid3)

        gm = GeofenceManager()
        # Adding invalid zone directly fails gracefully
        bad_zone = GeofenceZone(id="bad", name="Bad", polygon=[(10, 10), (20, 20)], enabled=True)
        res = gm.add_zone(bad_zone)
        self.assertFalse(res)
        self.assertEqual(len(gm.zones), 0)

        # Loading from config file
        config_file = os.path.join(PROJECT_ROOT, "config", "geofences.json")
        self.assertTrue(os.path.exists(config_file))
        res_load = gm.load_config(config_file)
        self.assertTrue(res_load)
        self.assertGreater(len(gm.zones), 0)

    def test_11_stale_track_cleanup_removes_state(self):
        """Scenario 11: Stale-track cleanup removes the correct track's geofence state."""
        gm = GeofenceManager(zones=[self.zone1])
        # Two tracks
        gm.update_track(track_id=1, center_point=(50, 50), label="F-16", frame_idx=1)
        gm.update_track(track_id=2, center_point=(50, 50), label="Su-27", frame_idx=1)
        self.assertIn(1, gm.track_states)
        self.assertIn(2, gm.track_states)

        # Cleanup track 1 as stale
        removed = gm.cleanup_stale(stale_track_ids=[1])
        self.assertEqual(removed, [1])
        self.assertNotIn(1, gm.track_states)
        self.assertIn(2, gm.track_states)

        # Active track cleanup alternative
        gm.update_track(track_id=3, center_point=(50, 50), label="MiG-29", frame_idx=1)
        # Active is only [3], so 2 should be purged
        removed2 = gm.cleanup_stale(active_track_ids=[3])
        self.assertEqual(removed2, [2])
        self.assertNotIn(2, gm.track_states)
        self.assertIn(3, gm.track_states)

    def test_12_reset_clears_transition_history(self):
        """Scenario 12: Resetting the session clears transition history."""
        gm = GeofenceManager(zones=[self.zone1])
        # Generate alert
        gm.update_track(track_id=1, center_point=(50, 50), label="F-16", frame_idx=1)
        gm.update_track(track_id=1, center_point=(200, 200), label="F-16", frame_idx=2)
        self.assertEqual(len(gm.alerts_history), 1)
        self.assertIn(1, gm.track_states)

        # Reset
        gm.reset()
        self.assertEqual(len(gm.alerts_history), 0)
        self.assertEqual(len(gm.track_states), 0)

        # If track 1 is updated inside now, it should behave as a fresh session (first observed inside -> no alert)
        alerts_after_reset = gm.update_track(track_id=1, center_point=(200, 200), label="F-16", frame_idx=3)
        self.assertEqual(len(alerts_after_reset), 0)
        self.assertEqual(len(gm.alerts_history), 0)

    def test_boundary_treatment_is_consistently_inside(self):
        """Verify that points on vertices and edges are consistently treated as inside."""
        poly = [(100.0, 100.0), (300.0, 100.0), (300.0, 300.0), (100.0, 300.0)]
        # Vertex
        self.assertTrue(point_in_polygon((100.0, 100.0), poly))
        # Edge midpoint
        self.assertTrue(point_in_polygon((200.0, 100.0), poly))
        # Interior
        self.assertTrue(point_in_polygon((200.0, 200.0), poly))
        # Outside
        self.assertFalse(point_in_polygon((99.9, 100.0), poly))
        self.assertFalse(point_in_polygon((50.0, 50.0), poly))


if __name__ == "__main__":
    unittest.main()
