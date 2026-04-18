from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import heapq
from datetime import datetime, timedelta

app = FastAPI()

# ============================
# REQUEST MODEL
# ============================
class RouteRequest(BaseModel):
    source: int
    target: int


# ============================
# 1. HAVERSINE + MIDPOINT
# ============================
def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def midpoint(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    bx = np.cos(lat2) * np.cos(lon2 - lon1)
    by = np.cos(lat2) * np.sin(lon2 - lon1)

    lat3 = np.arctan2(
        np.sin(lat1) + np.sin(lat2),
        np.sqrt((np.cos(lat1) + bx)**2 + by**2)
    )
    lon3 = lon1 + np.arctan2(by, np.cos(lat1) + bx)

    return np.degrees(lat3), np.degrees(lon3)


# ============================
# 2. FILTER NODES
# ============================
def find_nodes_between_places(csv_path, place1_id, place2_id):
    df = pd.read_csv(csv_path)

    p1 = df[df['placeId'] == place1_id].iloc[0]
    p2 = df[df['placeId'] == place2_id].iloc[0]

    lat1, lon1 = p1['latitude'], p1['longitude']
    lat2, lon2 = p2['latitude'], p2['longitude']

    distance = haversine_vectorized(lat1, lon1, lat2, lon2)
    mid_lat, mid_lon = midpoint(lat1, lon1, lat2, lon2)

    radius = 1.2 * distance

    df['dist_from_center'] = haversine_vectorized(
        mid_lat, mid_lon,
        df['latitude'].values,
        df['longitude'].values
    )

    nodes = df[df['dist_from_center'] <= radius]['placeId'].tolist()
    return nodes


# ============================
# 3. EDGE PREPROCESSING
# ============================
def preprocess_edge(df):
    df = pd.DataFrame(df)

    df['serviceStartTime'] = pd.to_datetime(df['serviceStartTime'], format='%H:%M')
    df['serviceEndTime'] = pd.to_datetime(df['serviceEndTime'], format='%H:%M')

    today = datetime.now().date()
    tomorrow = today + timedelta(days=1)

    df_today = df.copy()
    df_today['date'] = today

    df_tomorrow = df.copy()
    df_tomorrow['date'] = tomorrow

    df = pd.concat([df_today, df_tomorrow], ignore_index=True)

    df['start_datetime'] = pd.to_datetime(
        df['date'].astype(str) + ' ' + df['serviceStartTime'].dt.strftime('%H:%M')
    )

    df['end_datetime'] = pd.to_datetime(
        df['date'].astype(str) + ' ' + df['serviceEndTime'].dt.strftime('%H:%M')
    )

    df.loc[df['end_datetime'] < df['start_datetime'], 'end_datetime'] += timedelta(days=1)

    return df.sort_values(by='start_datetime').reset_index(drop=True)


def build_edge_schedules(graph):
    edge_schedules = {}

    for u, v, data in graph.edges(data=True):
        try:
            df = preprocess_edge(data['weight'])
            edge_schedules[(u, v)] = df
        except Exception:
            continue

    return edge_schedules


# ============================
# 4. TIME-DEPENDENT DIJKSTRA (FIXED)
# ============================
def time_dependent_dijkstra(graph, edge_schedules, source, target, buffer_minutes=30):
    start_time = datetime.now()

    heap = [(start_time, source, [])]

    best_time = {node: datetime.max for node in graph.nodes}
    best_time[source] = start_time

    visited = set()

    while heap:
        curr_time, node, path = heapq.heappop(heap)

        if node in visited:
            continue
        visited.add(node)

        if node == target:
            return curr_time, path

        for neighbor in graph.successors(node):
            if (node, neighbor) not in edge_schedules:
                continue

            df = edge_schedules[(node, neighbor)]
            earliest_board = curr_time + timedelta(minutes=buffer_minutes)

            idx = df['start_datetime'].searchsorted(earliest_board)
            if idx >= len(df):
                continue

            bus = df.iloc[idx]
            arrival_time = bus['end_datetime']

            if arrival_time < best_time.get(neighbor, datetime.max):
                best_time[neighbor] = arrival_time

                # ✅ store structured edge info
                step = {
                    "from": int(node),
                    "to": int(neighbor),
                    "bus": bus.to_dict()
                }

                heapq.heappush(heap, (arrival_time, neighbor, path + [step]))

    return None, None


# ============================
# LOAD DATA
# ============================
GRAPH_PATH = "apsrtc_directed_time_table_weighted_graph.pkl"
CSV_PATH = "data.csv"

with open(GRAPH_PATH, 'rb') as f:
    GRAPH = pickle.load(f)


# ============================
# API ENDPOINT
# ============================
@app.get("/route")
def route(source: int, target: int):

    valid_nodes = find_nodes_between_places(CSV_PATH, source, target)
    valid_nodes = set(valid_nodes)
    valid_nodes.add(source)
    valid_nodes.add(target)

    subgraph = GRAPH.subgraph(valid_nodes).copy()
    edge_schedules = build_edge_schedules(subgraph)

    arrival, path = time_dependent_dijkstra(
        subgraph,
        edge_schedules,
        source,
        target
    )

    if not arrival:
        return {"error": "No route found"}

    cleaned_path = []
    for step in path:
        bus = {
            k: (int(v) if hasattr(v, "item") else str(v))
            for k, v in step["bus"].items()
        }

        cleaned_path.append({
            "from": step["from"],
            "to": step["to"],
            "bus": bus
        })

    return {
        "arrival_time": str(arrival),
        "best_path": cleaned_path,
        "all_possible_times": str(arrival),
        "all_possible_paths": cleaned_path
    }