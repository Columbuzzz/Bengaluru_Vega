#!/usr/bin/env python3
import os
import re
import json
import argparse
import warnings
from datetime import timedelta

import pandas as pd
import numpy as np
import joblib
import pickle
import h3
import networkx as nx
import osmnx as ox
from geopy.distance import geodesic
from shapely.geometry import Point, Polygon, box
from shapely.ops import transform
from pyproj import CRS, Transformer
import math

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------
# Path and Config Setup
# ---------------------------
IN_DOCKER = os.path.exists("/app")
BASE_PATH = "/app" if IN_DOCKER else "."
REF_PATH = os.path.join(BASE_PATH, "ref_data")

# --- Task 1 (Bus) Paths ---
STOPS_PATH = os.path.join(REF_PATH, "stops_clean.csv")
ROUTE_SEQ_PATH = os.path.join(REF_PATH, "route_to_stop_clean.csv")
MODEL_PATH = os.path.join(REF_PATH, "models/eta_model_hypertuned.pkl")
ENCODERS_PATH = os.path.join(REF_PATH, "models/encoders_hypertuned.pkl")
SPEED_PATH = os.path.join(REF_PATH, "models/route_avg_speed_m_s_hypertuned.pkl")

# --- Task 2 (Auto) Paths ---
SDR_PATH = os.path.join(REF_PATH, "processed_hex_avg_SDR.parquet")
AUTO_SPEED_PATH = os.path.join(REF_PATH, "smoothed_speed_full.parquet")
GRAPH_PATH = os.path.join(REF_PATH, "bangalore_graph_99.6.graphml")

# --- Task 2 Constants (Updated/Confirmed) ---
H3_RES = 7
CIRCLE_RADIUS_M = 1840.0
PA_ALPHA, PA_BETA = 2.0, 0.0
PB_ALPHA, PB_BETA = 0.2, 0
PC_CONST = 10.0 # Fallback trip duration in minutes
SDR_FAILURE_THRESHOLD = 0.1
DEFAULT_AUTO_SPEED_KPH = 19.489212994167335 # Used as default speed in PC calculation

# ---------------------------
# Helper Functions (Consolidated and made robust)
# ---------------------------
def parse_coordinates(s):
    if pd.isna(s): return (None, None)
    try:
        # Robust parsing from the second script
        parts = [float(p.strip()) for p in str(s).strip().replace("(", "").replace(")", "").replace('"', '').split(",")]
        # Latitude is first, Longitude is second
        return (parts[0], parts[1]) if len(parts) == 2 else (None, None)
    except:
        return (None, None)

def safe_read_parquet(path):
    if not os.path.exists(path):
        alt_path = os.path.join("/app/data", path)
        if os.path.exists(alt_path):
            return pd.read_parquet(alt_path)
        # Note: In a docker environment, files might be in /app/ref_data if path is ref_data/...
        if IN_DOCKER and not os.path.isabs(path):
            alt_path_2 = os.path.join("/app", path)
            if os.path.exists(alt_path_2):
                return pd.read_parquet(alt_path_2)

        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_parquet(path)

def build_aeqd_transformer(lat_center, lon_center):
    aeqd_proj4 = f"+proj=aeqd +lat_0={lat_center} +lon_0={lon_center} +units=m +datum=WGS84 +no_defs"
    aeqd_crs = CRS.from_proj4(aeqd_proj4)
    wgs84 = CRS.from_epsg(4326)
    to_proj = Transformer.from_crs(wgs84, aeqd_crs, always_xy=True)
    return to_proj.transform, None # Only need to_proj for circle intersection

def get_h3_index(lat, lon, res):
    """Robustly gets the H3 index as a HEX STRING."""
    try:
        return h3.latlng_to_cell(lat, lon, res)
    except Exception:
        return None

def convert_h3_to_int64(h3_hex_str):
    """Converts a hexadecimal H3 string to a 64-bit integer, safely handling invalid inputs."""
    if pd.isna(h3_hex_str) or h3_hex_str is None:
        return np.int64(0)
    try:
        return np.int64(int(str(h3_hex_str), 16))
    except ValueError:
        return np.int64(0)

# ---------------------------
# Task 2: Auto Time Predictor Class
# ---------------------------
class AutoTimePredictor:
    def __init__(self):
        print("[Auto] Initializing AutoTimePredictor...")
        
        # --- SDR Data Setup ---
        sdr_df = safe_read_parquet(SDR_PATH)
        sdr_long_df = sdr_df.melt(id_vars=['h3_index'], value_vars=[f'slot_{i}' for i in range(96)], var_name='slot_name', value_name='SDR')
        sdr_long_df['slot'] = sdr_long_df['slot_name'].str.split('_').str[-1].astype(int)
        sdr_long_df['h3_index'] = sdr_long_df['h3_index'].astype(np.int64)
        self.sdr_series = sdr_long_df.set_index(["h3_index", "slot"])["SDR"]
        
        # --- Speed Data Setup ---
        self.speed_df = safe_read_parquet(AUTO_SPEED_PATH)
        self.speed_df.columns = [str(col) for col in self.speed_df.columns]
        if 'u' in self.speed_df.columns: self.speed_df['u'] = self.speed_df['u'].astype(np.int64)
        if 'v' in self.speed_df.columns: self.speed_df['v'] = self.speed_df['v'].astype(np.int64)
        self.default_speed_kph = np.float64(DEFAULT_AUTO_SPEED_KPH)
        
        # The original code loaded the graph, keep a placeholder but the new PC logic doesn't use it.
        self.G = None
        # try:
        #     self.G = ox.load_graphml(GRAPH_PATH)
        # except Exception:
        #     pass

        print("[Auto] AutoTimePredictor ready.")

    def _get_h3_index(self, lat, lon):
        return get_h3_index(lat, lon, H3_RES)

    def _compute_hex_circle_weights(self, lat, lon, k_ring_k=3):
        # Implementation from the new robust logic
        center_h3_hexadecimal = self._get_h3_index(lat, lon)
        
        if center_h3_hexadecimal is None or center_h3_hexadecimal == '0':
            return []
            
        center_h3_int64 = convert_h3_to_int64(center_h3_hexadecimal)

        if center_h3_int64 == np.int64(0):
            return []

        candidates = h3.grid_disk(center_h3_hexadecimal, k_ring_k)

        to_proj, _ = build_aeqd_transformer(lat, lon)
        cx, cy = to_proj(lon, lat)
        circle_proj = Point(cx, cy).buffer(CIRCLE_RADIUS_M, resolution=64)

        weights = []
        for h_hex_str in candidates: 
            h3_index_int64 = convert_h3_to_int64(h_hex_str) 

            try:
                boundary_func = h3.cell_to_boundary
            except AttributeError:
                # Fallback for older h3 library versions
                boundary_func = h3.h3_to_geo_boundary

            # Note: h3 boundary returns (lat, lng), Polygon needs (lng, lat)
            hex_coords = [(lng, lat) for lat, lng in boundary_func(h_hex_str)] 
            hex_poly_wgs = Polygon(hex_coords)
            hex_poly_proj = transform(lambda x, y: to_proj(x, y), hex_poly_wgs)

            inter = hex_poly_proj.intersection(circle_proj)
            inter_area = inter.area if not inter.is_empty else 0.0
            hex_area = hex_poly_proj.area if hex_poly_proj.area > 0 else 1.0
            weight = inter_area / hex_area

            if weight > 0:
                weights.append((h3_index_int64, weight))
                
        return weights

    # --- UPDATED: New logic for PA/PB calculation ---
    def _predict_pa_pb(self, lat, lon, req_time_ist):
        slot = min(max((req_time_ist.hour * 60 + req_time_ist.minute) // 15, 0), 95)
        weights = self._compute_hex_circle_weights(lat, lon)
        
        if not weights:
            print(" -> CHECKPOINT: No H3 neighbors found for auto PA/PB. Using center hex fallback.")
            center_h3_hex = self._get_h3_index(lat, lon)
            center_h3_int = convert_h3_to_int64(center_h3_hex)
            if center_h3_int != np.int64(0):
                weights = [(center_h3_int, 1.0)]
            else:
                return 30.0, 1.0 # Absolute fallback

        SDRs, areas = [], []
        for h, w in weights:
            # Use .get for safe lookup
            SDRs.append(self.sdr_series.get((h, slot), 0.0))
            areas.append(w)

        SDR_list = np.array(SDRs)
        area_list = np.array(areas)
        weighted_sdr = np.sum(SDR_list * area_list)
        
        # PA Calculation (Wait Time)
        if weighted_sdr < SDR_FAILURE_THRESHOLD:
            pa = 30.0
        else:
            log_transformed_sdr = math.log(1 + weighted_sdr)
            y_pa = PA_ALPHA * log_transformed_sdr + PA_BETA
            prob_pa = 1 / (1 + np.exp(-y_pa))
            pa = 9.0 * (1 - prob_pa) 
        
        # PB Calculation (Negotiation Time)
        y_pb = PB_ALPHA * weighted_sdr + PB_BETA
        prob_pb = 1 / (1 + np.exp(-y_pb))
        pb = 7 * (1 - prob_pb)
            
        return pa, pb

    # --- HELPER for Dynamic PC ---
    def _edge_speed_in_df(self, u, v, timeslot_col, G):
        def mean_speed(df):
            # Fallback to the default speed if no data is found in the slice
            return float(df[timeslot_col].mean()) if not df.empty and df[timeslot_col].mean() > 0.5 else np.nan

        # 1. EXACT MATCHES (u->v or v->u)
        match = self.speed_df[((self.speed_df['u'] == u) & (self.speed_df['v'] == v)) | ((self.speed_df['u'] == v) & (self.speed_df['v'] == u))]
        speed = mean_speed(match)
        if not pd.isna(speed): return speed

        # 2. OSMID MATCH (Slower, but more robust)
        edge_data_dict = G.get_edge_data(u, v)
        osmid_set = set()
        if edge_data_dict:
            for _, attr in edge_data_dict.items():
                osmid = attr.get('osmid')
                if osmid is None: continue
                if isinstance(osmid, list): osmid_set.update(osmid)
                else: osmid_set.add(osmid)
        if osmid_set:
            match = self.speed_df[self.speed_df['osmid'].isin(osmid_set)]
            speed = mean_speed(match)
            if not pd.isna(speed): return speed

        # 3. PARTIAL/INFERRED MATCHES (any edge connected to u or v)
        match = self.speed_df[(self.speed_df['u'] == u) | (self.speed_df['v'] == v) | (self.speed_df['u'] == v) | (self.speed_df['v'] == u)]
        speed = mean_speed(match)
        if not pd.isna(speed): return speed
        
        return np.nan # Return NaN if no speed is found

    # --- UPDATED: New logic for PC calculation (Dynamic Graph) ---
    def _predict_pc(self, start_loc, end_loc, req_time_ist):
        start_lat, start_lon = start_loc[0], start_loc[1]
        end_lat, end_lon = end_loc[0], end_loc[1]
        time_str = req_time_ist.strftime("%H:%M:%S")

        try:
            hh, mm, ss = [int(x) for x in time_str.split(":")]
            seconds_from_midnight = hh * 3600 + mm * 60 + ss
            timeslot = str(seconds_from_midnight // 900)

            buffer_deg = 0.01
            north, south = max(start_lat, end_lat) + buffer_deg, min(start_lat, end_lat) - buffer_deg
            east, west = max(start_lon, end_lon) + buffer_deg, min(start_lon, end_lon) - buffer_deg
            bbox_polygon = box(west, south, east, north)
            
            # Dynamically get the graph for the route bounding box (may take time)
            G = ox.graph_from_polygon(bbox_polygon, network_type="drive_service", simplify=True)

            # Get nearest nodes. Note: osmnx uses (longitude, latitude)
            u_node = ox.nearest_nodes(G, start_lon, start_lat)
            v_node = ox.nearest_nodes(G, end_lon, end_lat)
            
            route_nodes = nx.shortest_path(G, source=u_node, target=v_node, weight='length')
            
            total_time_sec = 0
            last_speed = self.default_speed_kph
            
            for i in range(len(route_nodes) - 1):
                u, v = route_nodes[i], route_nodes[i + 1]
                avg_speed = self._edge_speed_in_df(u, v, timeslot, G)
                
                # Use last speed or default speed if speed is not found (NaN)
                if pd.isna(avg_speed):
                    avg_speed = last_speed
                
                last_speed = avg_speed
                
                edge_data = G.get_edge_data(u, v, key=0) 
                edge_length_m = edge_data.get('length', 0)
                
                if avg_speed > 0:
                    time_sec = (edge_length_m / 1000) / avg_speed * 3600
                    total_time_sec += time_sec
            
            return total_time_sec / 60 # Return in minutes
        
        except Exception as e:
            # CHECKPOINT
            print(f" -> CHECKPOINT: Auto PC dynamic route calculation failed. Using fallback value {PC_CONST}. Error: {e}")
            return PC_CONST

    def predict_trip(self, start_loc, end_loc, req_time_ist):
        pa, pb = self._predict_pa_pb(start_loc[0], start_loc[1], req_time_ist)
        pc = self._predict_pc(start_loc, end_loc, req_time_ist)
        return pa, pb, pc

# ---------------------------
# Task 1: Bus ETA Predictor Class (MODIFIED)
# ---------------------------
class BusETAPredictor:
    # <<< MODIFIED: Simplified to accept only one default buffer time
    def __init__(self, default_buffer_time_mins=0.5):
        print("[Bus] Initializing BusETAPredictor...")
        self.model = joblib.load(MODEL_PATH)
        with open(ENCODERS_PATH, "rb") as f:
            self.encoders = pickle.load(f)
        self.route_avg_speed = joblib.load(SPEED_PATH)
        self.global_avg_speed = float(pd.Series(self.route_avg_speed).mean())
        
        self.df_stops = pd.read_csv(STOPS_PATH)
        self.df_route_seq = pd.read_csv(ROUTE_SEQ_PATH)
        self.df_route_seq['route_id'] = self.df_route_seq['route_id'].astype(str)

        # <<< MODIFIED: Store the single buffer time configuration
        self.default_buffer_time_mins = default_buffer_time_mins
        print(f"[Bus] Dwell time configured: Using a uniform buffer of {self.default_buffer_time_mins} mins for all stops.")

        print("[Bus] BusETAPredictor ready.")

    def get_stop_coords(self, stop_id):
        stop_info = self.df_stops[self.df_stops['stop_id'] == stop_id]
        if not stop_info.empty:
            return (stop_info.iloc[0]['stop_lat'], stop_info.iloc[0]['stop_lon'])
        return (None, None)

    def _parse_stop_list(self, stop_list_str):
        if pd.isna(stop_list_str): return []
        return [int(n) for n in re.findall(r"\d+", str(stop_list_str))]

    def _predict_segment_duration_mins(self, route_id, from_stop_id, to_stop_id, distance_m, timestamp_utc):
        r_enc = self.encoders['route_id'].transform([str(route_id)])[0] if str(route_id) in self.encoders['route_id'].classes_ else 0
        fs_enc = self.encoders['from_stop'].transform([str(from_stop_id)])[0] if str(from_stop_id) in self.encoders['from_stop'].classes_ else 0
        ts_enc = self.encoders['to_stop'].transform([str(to_stop_id)])[0] if str(to_stop_id) in self.encoders['to_stop'].classes_ else 0
        avg_speed = self.route_avg_speed.get(str(route_id), self.global_avg_speed)

        features = pd.DataFrame([{
            'route_id_enc': r_enc, 'from_stop_enc': fs_enc, 'to_stop_enc': ts_enc,
            'distance_m': distance_m, 'avg_speed_m_s': avg_speed,
            'start_hour': int(timestamp_utc.hour), 'day_of_week': int(timestamp_utc.dayofweek)
        }])
        return float(np.expm1(self.model.predict(features)[0]))

    # <<< MODIFIED: This method now includes the default buffer time
    def predict_eta_to_stop(self, last_ping, route_id, target_stop_id):
        ping_time = pd.to_datetime(last_ping['vehicle_timestamp'], unit='s', utc=True)
        ping_loc = (last_ping['latitude'], last_ping['longitude'])
        
        route_info = self.df_route_seq[self.df_route_seq['route_id'] == str(route_id)]
        if route_info.empty: return None
        
        stop_seq = self._parse_stop_list(route_info['stop_id_list'].iloc[0])
        if not stop_seq: return None
        
        stop_coords = self.df_stops.set_index('stop_id').loc[stop_seq][['stop_lat', 'stop_lon']]
        dists = stop_coords.apply(lambda r: geodesic(ping_loc, (r.stop_lat, r.stop_lon)).meters, axis=1)
        next_idx = dists.idxmin()
        
        try:
            target_seq_idx = stop_seq.index(target_stop_id)
            next_seq_idx = stop_seq.index(next_idx)
        except ValueError:
            return None # Target stop not in sequence

        if target_seq_idx < next_seq_idx: return None # Bus has passed the target stop

        cumulative_mins = 0.0
        
        # From ping to first stop
        first_stop_coords = self.get_stop_coords(next_idx)
        dist_to_first_stop = geodesic(ping_loc, first_stop_coords).meters
        from_stop_id = stop_seq[next_seq_idx-1] if next_seq_idx > 0 else next_idx
        cumulative_mins += self._predict_segment_duration_mins(route_id, from_stop_id, next_idx, dist_to_first_stop, ping_time)

        if target_stop_id == next_idx:
            return ping_time + timedelta(minutes=cumulative_mins)
            
        # From first stop to subsequent stops
        for i in range(next_seq_idx, target_seq_idx):
            f_stop, t_stop = stop_seq[i], stop_seq[i+1]
            f_loc, t_loc = self.get_stop_coords(f_stop), self.get_stop_coords(t_stop)
            dist_m = geodesic(f_loc, t_loc).meters
            
            # Add travel time for the segment
            cumulative_mins += self._predict_segment_duration_mins(route_id, f_stop, t_stop, dist_m, ping_time)
            
            # <<< MODIFIED: Add default buffer time for the stop just arrived at, unless it's the final target stop
            if t_stop != target_stop_id:
                cumulative_mins += self.default_buffer_time_mins
            
        return ping_time + timedelta(minutes=cumulative_mins)

    # <<< MODIFIED: This method now includes the default buffer time
    def predict_trip_duration(self, route_id, start_stop_id, end_stop_id, departure_time_utc):
        route_info = self.df_route_seq[self.df_route_seq['route_id'] == str(route_id)]
        if route_info.empty:
            # CHECKPOINT
            print(f" -> CHECKPOINT: Bus route {route_id} not found for trip duration. Using fallback.")
            return 15.0 
        
        stop_seq = self._parse_stop_list(route_info['stop_id_list'].iloc[0])
        try:
            start_idx = stop_seq.index(start_stop_id)
            end_idx = stop_seq.index(end_stop_id)
        except ValueError:
            # CHECKPOINT
            print(f" -> CHECKPOINT: Start/end stops not in sequence for route {route_id}. Using fallback.")
            return 15.0 

        if start_idx >= end_idx:
            # CHECKPOINT
            print(f" -> CHECKPOINT: Start stop is after end stop for route {route_id}. Using fallback.")
            return 5.0 
        
        total_duration_mins = 0.0
        for i in range(start_idx, end_idx):
            f_stop, t_stop = stop_seq[i], stop_seq[i+1]
            f_loc, t_loc = self.get_stop_coords(f_stop), self.get_stop_coords(t_stop)
            dist_m = geodesic(f_loc, t_loc).meters
            
            # Add travel time for the segment
            total_duration_mins += self._predict_segment_duration_mins(route_id, f_stop, t_stop, dist_m, departure_time_utc)

            # <<< MODIFIED: Add default buffer time for each intermediate stop the passenger arrives at.
            # Do not add buffer for the final stop where they disembark.
            if t_stop != end_stop_id:
                total_duration_mins += self.default_buffer_time_mins
            
        return total_duration_mins

# ---------------------------
# Main Execution Logic (Corrected)
# ---------------------------
def main(input_csv_path, output_json_path):
    print("--- Starting Bengaluru Last Mile Challenge - Task 3 ---")
    
    # <<< MODIFIED: TUNEABLE PARAMETER FOR BUS STOP BUFFER >>>
    # This is the uniform buffer time in minutes for any intermediate stop.
    # 0.75 minutes = 45 seconds.
    default_buffer_in_mins = 0.0
    # <<< END OF TUNEABLE PARAMETER >>>

    auto_predictor = AutoTimePredictor()
    # <<< MODIFIED: Pass the single buffer configuration to the predictor
    bus_predictor = BusETAPredictor(
        default_buffer_time_mins=default_buffer_in_mins
    )

    # NOTE: pd.read_csv should be robust to the path if safe_read_parquet works well.
    df_journeys = pd.read_csv(input_csv_path)
    output_results = {}

    for _, journey in df_journeys.iterrows():
        jid = journey['jid']
        print(f"\n--- Processing Journey ID: {jid} ---")
        try:
            # --- FIX: Read bus data first to get the correct date ---
            bus_data_path = journey['path_to_the_parquet_file']
            df_bus_live = safe_read_parquet(bus_data_path)
            
            # Extract date from the first timestamp in the bus data
            bus_timestamp = pd.to_datetime(df_bus_live['vehicle_timestamp'].iloc[0], unit='s', utc=True)
            journey_date_str = bus_timestamp.strftime('%Y-%m-%d')

            # --- Now construct the request time with the CORRECT date ---
            start_loc = parse_coordinates(journey['auto_1_ride_start_location'])
            req_time_ist = pd.to_datetime(f"{journey_date_str} {journey['auto_1_ride_request_time']}").tz_localize('Asia/Kolkata')
            boarding_stop_id = int(journey['auto_1_ride_end_bus_stop_ID'])
            disembark_stop_id = int(journey['bus_trip_end_bus_stop_ID'])
            final_dest_loc = parse_coordinates(journey['auto_2_ride_end_location'])

            # --- 1. First Auto Ride ---
            print("[CHECKPOINT] Predicting First Auto Ride (a1, a2, a3)...")
            boarding_stop_loc = bus_predictor.get_stop_coords(boarding_stop_id)
            a1, a2, a3 = auto_predictor.predict_trip(start_loc, boarding_stop_loc, req_time_ist)
            a3_tag = "(Fallback Used)" if a3 == PC_CONST else "(Model Prediction)"
            print(f" -> SUCCESS: a1 = {a1:.2f}, a2 = {a2:.2f} (Model Prediction)")
            print(f" -> SUCCESS: a3 = {a3:.2f} {a3_tag}")


            # --- 2. Arrival at Bus Stop ---
            arrival_at_bus_stop_ist = req_time_ist + timedelta(minutes=(a1 + a2 + a3))
            arrival_at_bus_stop_utc = arrival_at_bus_stop_ist.tz_convert('UTC')
            print(f" -> INFO: Passenger arrival at bus stop (UTC): {arrival_at_bus_stop_utc.strftime('%Y-%m-%d %H:%M:%S')}")

            # --- 3. Bus Leg (a4, a5) ---
            print("[CHECKPOINT] Predicting Bus Wait Time (a4) and Journey Time (a5)...")
            route_id = df_bus_live['route_id'].mode()[0]
            
            future_bus_etas = {}
            print(" -> INFO: Checking ETAs for available buses...")
            for trip_id in df_bus_live['trip_id'].unique():
                df_trip = df_bus_live[df_bus_live['trip_id'] == trip_id]
                last_ping = df_trip.sort_values('vehicle_timestamp').iloc[-1]
                eta_utc = bus_predictor.predict_eta_to_stop(last_ping, route_id, boarding_stop_id)
                if eta_utc:
                    print(f"     - Bus (trip_id {trip_id}) ETA: {eta_utc.strftime('%Y-%m-%d %H:%M:%S')}")
                    if eta_utc > arrival_at_bus_stop_utc:
                        future_bus_etas[trip_id] = eta_utc
            
            a4_tag = "(Model Prediction)"
            if not future_bus_etas:
                a4_tag = "(Fallback Used)"
                print(f" -> FALLBACK: No upcoming buses found. Using 15 min fallback wait time for a4.")
                actual_boarding_time_utc = arrival_at_bus_stop_utc + timedelta(minutes=15)
            else:
                best_trip_id, actual_boarding_time_utc = min(future_bus_etas.items(), key=lambda item: item[1])
                print(f" -> SUCCESS: Catching bus (trip_id {best_trip_id}) arriving at {actual_boarding_time_utc.strftime('%Y-%m-%d %H:%M:%S')}")

            a4 = max(0, (actual_boarding_time_utc - arrival_at_bus_stop_utc).total_seconds() / 60)
            print(f" -> SUCCESS: a4 = {a4:.2f} {a4_tag}")

            a5 = bus_predictor.predict_trip_duration(route_id, boarding_stop_id, disembark_stop_id, actual_boarding_time_utc)
            a5_tag = "(Fallback Used)" if a5 in [5.0, 15.0] else "(Model Prediction)"
            print(f" -> SUCCESS: a5 = {a5:.2f} {a5_tag}")


            # --- 4. Arrival at Destination Stop ---
            arrival_at_dest_stop_utc = actual_boarding_time_utc + timedelta(minutes=a5)
            arrival_at_dest_stop_ist = arrival_at_dest_stop_utc.tz_convert('Asia/Kolkata')

            # --- 5. Second Auto Ride ---
            print("[CHECKPOINT] Predicting Second Auto Ride (a6, a7, a8)...")
            disembark_stop_loc = bus_predictor.get_stop_coords(disembark_stop_id)
            a6, a7, a8 = auto_predictor.predict_trip(disembark_stop_loc, final_dest_loc, arrival_at_dest_stop_ist)
            a8_tag = "(Fallback Used)" if a8 == PC_CONST else "(Model Prediction)"
            print(f" -> SUCCESS: a6 = {a6:.2f}, a7 = {a7:.2f} (Model Prediction)")
            print(f" -> SUCCESS: a8 = {a8:.2f} {a8_tag}")
            
            # --- Store Results ---
            final_predictions = {
                "a1": round(a1, 4), "a2": round(a2, 4), "a3": round(a3, 4),
                "a4": round(a4, 4), "a5": round(a5, 4), "a6": round(a6, 4),
                "a7": round(a7, 4), "a8": round(a8, 4)
            }
            output_results[str(jid)] = final_predictions
            print(f" -> FINAL PREDICTIONS for Journey {jid}: {final_predictions}")

        except Exception as e:
            print(f" -> FATAL ERROR processing Journey ID {jid}: {e}")
            print(f" -> FALLBACK: Using default values for all 8 components for Journey {jid}.")
            output_results[str(jid)] = {
                "a1": 2.0, "a2": 3.0, "a3": 15.0, "a4": 10.0,
                "a5": 20.0, "a6": 2.0, "a7": 3.0, "a8": 15.0
            }
        print(f"--- Finished Journey ID: {jid} ---")


    # --- Save Final Output ---
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(output_results, f, indent=4)
    print(f"\n✅ Predictions complete. Output saved to {output_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 3 Multi-Modal Journey Time Prediction")
    parser.add_argument("--input_csv", required=True, help="Path to the input CSV file")
    parser.add_argument("--output_json", required=True, help="Path to write the output JSON file")
    args = parser.parse_args()
    
    main(args.input_csv, args.output_json)