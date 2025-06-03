
import os
import json
import time
import logging
import datetime
import pandas as pd
from tqdm import tqdm
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import matplotlib.pyplot as plt
from collections import Counter
from rapidfuzz import process, fuzz
import numpy as np

# --- Config ---
CLIENT_ID = "0882d646bde94b6f90d84fa91942a45b"
CLIENT_SECRET = "6f8d91a818e74a878ee5e356332a1bc9"
RAW_PATH = "C:/Users/alici/OneDrive/Desktop/Spotify_Streaming_History.json"
OUTPUT_PATH = "C:/Users/alici/OneDrive/Desktop/Spotify_Streaming_History_With_Genres.json"
GENRE_CACHE_PATH = "C:/Users/alici/OneDrive/Desktop/artist_genre_cache.json"
TRACK_ARTIST_MAP_PATH = "C:/Users/alici/OneDrive/Desktop/track_to_artists_cache.json"

# --- Logging ---
log_path = "C:/Users/alici/OneDrive/Desktop/spotify_genre_fetch.log"
logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info("Script started")

# --- Spotify API Setup ---
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(CLIENT_ID, CLIENT_SECRET), requests_timeout = 20)

# --- Load Raw data ---
df = pd.read_json(RAW_PATH)
# --- Load other persons Spotify Data ---

df_person_2 = pd.read_json("C:/Users/alici/OneDrive/Desktop/Chris_Streaming_History.json")


"""
Data Cleaning
"""

# --- For Main Data ---
df["ts"] = pd.to_datetime(df["ts"])
df["day"] = df["ts"].dt.date
df["hour"] = df["ts"].dt.time

df = df.drop("ts", axis = 1)

col_to_mv = df.pop("day")
df.insert(0, "day", col_to_mv)

col_to_mv2 = df.pop("hour")
df.insert(1, "hour", col_to_mv2)

df = df.rename(columns = {"day" : "Day_Song_Played", "hour" : "Time_of_Day_Played", 
                    "conn_country" : "Country_Song_Played",
                    "master_metadata_track_name" : "Song_Name", 
                    "master_metadata_album_artist_name" : "Artist_Name",
                    "master_metadata_album_album_name" : "Album_Name"})



# -- Data Cleaning for Person 2 ---
df_person_2["endTime"] = pd.to_datetime(df_person_2["endTime"])
df_person_2["Day_Song_Played"] = df_person_2["endTime"].dt.date
df_person_2["Time_of_Day_Played"] = df_person_2["endTime"].dt.time

df_person_2 = df_person_2.drop(["msPlayed","endTime"], axis = 1)

col_to_mv = df_person_2.pop("Day_Song_Played")
df_person_2.insert(0, "Day_Song_Played", col_to_mv)

col_to_mv2 = df_person_2.pop("Time_of_Day_Played")
df_person_2.insert(1, "Time_of_Day_Played", col_to_mv2)

df_person_2 = df_person_2.rename(columns = {"trackName" : "Song_Name", 
                    "artistName" : "Artist_Name"})


"""
Create New json File with genres
"""

# Extract track IDs
df["track_id"] = df["spotify_track_uri"].apply(lambda x: x.split(":")[-1] if isinstance(x, str) and "spotify:track:" in x else None)
df = df[df["track_id"].notnull()].reset_index(drop=True)

# --- Load or initialize caches ---
artist_genre_map = json.load(open(GENRE_CACHE_PATH)) if os.path.exists(GENRE_CACHE_PATH) else {}
track_to_artists = json.load(open(TRACK_ARTIST_MAP_PATH)) if os.path.exists(TRACK_ARTIST_MAP_PATH) else {}

# --- Fetch missing artist IDs ---
track_ids = df["track_id"].unique().tolist()
unknown_tracks = [tid for tid in track_ids if tid not in track_to_artists]
for i in tqdm(range(0, len(unknown_tracks), 50), desc="Track→Artist"):
    batch = unknown_tracks[i:i+50]
    data = sp.tracks(batch)["tracks"]
    for track in data:
        if track and track["id"]:
            track_to_artists[track["id"]] = [a["id"] for a in track["artists"]]
    time.sleep(0.2)
with open(TRACK_ARTIST_MAP_PATH, "w") as f:
    json.dump(track_to_artists, f)


# --- Fetch missing genres ---
all_artist_ids = set(aid for aids in track_to_artists.values() for aid in aids)
missing_artist_ids = [aid for aid in all_artist_ids if aid not in artist_genre_map]
for i in tqdm(range(0, len(missing_artist_ids), 50), desc="Fetching Genres"):
    batch = missing_artist_ids[i:i+50]
    artists_data = sp.artists(batch)["artists"]
    for artist in artists_data:
        if artist and artist["id"]:
            artist_genre_map[artist["id"]] = artist.get("genres", [])
    time.sleep(0.5)
with open(GENRE_CACHE_PATH, "w") as f:
    json.dump(artist_genre_map, f)

# --- Assign genres to each track ---
def get_genres(tid):
    genres = set()
    for aid in track_to_artists.get(tid, []):
        genres.update(artist_genre_map.get(aid, []))
    return list(genres)
df["genres"] = df["track_id"].apply(get_genres)


# --- Save enriched data ---
df.to_json(OUTPUT_PATH, orient = "records", indent = 2)
logging.info("Genres saved to output")



"""
Analyzing Data
"""

# --- Print and Count Repeated Artists ---
repeated_artists_count = df["Artist_Name"].value_counts()
repeated_artists = repeated_artists_count[repeated_artists_count >= 1]

if not repeated_artists.empty:
    print(f"Repeated Artists Count ('Artist_Name') : ")
    print(repeated_artists)

else:
    print(f"No repeated artists")


# --- Print and Count Repeated Songs ---
repeated_songs_count = df["Song_Name"].value_counts()
repeated_songs = repeated_songs_count[repeated_songs_count >= 1]


if not repeated_songs.empty:
    print(f"Repeated Song Count ('Song_Name') :" )
    print(repeated_songs)

else:
    print(f"No repeated songs")


"""
Creating Visuals For the Data
"""

 # --- Donut Chart of Top 10 Genres ---
def normalize_genre(genre):
    genre = genre.lower().strip().replace('-', ' ')
    # Remove common suffixes that often cause duplicates
    for suffix in [' beats', ' music', ' songs']:
        if genre.endswith(suffix):
            genre = genre[: -len(suffix)]
    return genre.strip()

def cluster_genres(raw_genres, threshold = 80):
    canonical = []
    clustered = []

    for genre in raw_genres:
        norm_genre = normalize_genre(genre)
        match = process.extractOne(norm_genre, canonical, scorer = fuzz.ratio)

        if match and match[1] >= threshold:
            clustered.append(match[0])
        else:
            canonical.append(norm_genre)
            clustered.append(norm_genre)

    return clustered

def plot_donut_chart(top_genres, title):
    labels = [genre for genre, count in top_genres]
    sizes = [count for genre, count in top_genres]

    custom_colors = ['#5F9EA0', '#4682B4', '#4169E1', '#87CEFA', '#B0C4DE',
                     '#E6E6FA', '#D8BFD8', '#FFF0F5', '#9370DB', '#483D8B']

    fig, ax = plt.subplots()
    wedges, texts = ax.pie(
        sizes,
        labels=labels,
        startangle=140,
        wedgeprops=dict(width = 0.3),
        colors=custom_colors,
      textprops = {'fontsize' : 9}
    )

    # Draw donut hole
    centre_circle = plt.Circle((0, 0), 0.70, fc = 'white')
    fig.gca().add_artist(centre_circle)

    total = sum(sizes)
    for i, p in enumerate(wedges):
        angle = (p.theta2 + p.theta1) / 2.0
        x = np.cos(np.deg2rad(angle)) * 0.85
        y = np.sin(np.deg2rad(angle)) * 0.85
        percentage = f"{(sizes[i] / total) * 100:.1f}%"
        ax.text(x, y, percentage, ha = 'center', va = 'center', fontsize = 8, color = 'black')

    plt.title(title, y = 1.05)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

df_exploded = df.explode('genres')
df_exploded = df_exploded[df_exploded['genres'].notnull() & (df_exploded['genres'] != '')]

# --- Overall Genre Plot ---
all_genres = df_exploded['genres'].tolist()
clustered_genres = cluster_genres(all_genres, threshold = 80)
counter = Counter(clustered_genres)
top_10_overall = counter.most_common(10)
plot_donut_chart(top_10_overall, "Top 10 Genres Overall")



# --- Genre Donut Chart Per Year ---
df["Day_Song_Played"] = pd.to_datetime(df["Day_Song_Played"])
df['year'] = df['Day_Song_Played'].dt.year

df_exploded = df.explode('genres')
df_exploded = df_exploded[df_exploded['genres'].notnull() & (df_exploded['genres'] != '')]

# --- Year-specific Plot ---
available_years = sorted(df_exploded['year'].unique())
while True:
    try:
        selected_year = int(input(f"Enter a year to view between 2019 - 2025: "))
        if selected_year in available_years:
            break
        else:
            print("That year is not available in the data.")
    except ValueError:
        print("Please enter a valid year.")

year_data = df_exploded[df_exploded['year'] == selected_year]
year_genres = year_data['genres'].tolist()

clustered_year_genres = cluster_genres(year_genres, threshold=80)

year_counter = Counter(clustered_year_genres)
top_10_year = year_counter.most_common(10)

plot_donut_chart(top_10_year, f"Top 10 Genres in {selected_year}")



# --- Complete Graph of Songs Listened ---
count_songs_per_day = df['Day_Song_Played'].value_counts().sort_index()

plt.figure(figsize=(10,6))
plt.scatter(count_songs_per_day.index, count_songs_per_day.values, s = 6)

plt.xlabel("Year")
plt.ylabel("Song Count")
plt.title("Songs Listened To From 2019 May 17th - 2025 May 6th")
plt.show()



# --- Single Days Worth of Songs ---
user_input = input("Enter a date between 2019-04-17 and 2025-05-06 to look at (YYYY-MM-DD): ")
selected_date = pd.to_datetime(user_input).date()


df_day = df[df['Day_Song_Played'].dt.date == selected_date].copy()

df_day["Time_of_Day_Played"] = pd.to_datetime(df_day["Time_of_Day_Played"], format = "%H:%M:%S")

df_day['hour'] = df_day['Time_of_Day_Played'].dt.hour

songs_per_hour = df_day['hour'].value_counts().sort_index()
songs_per_hour = songs_per_hour.reindex(range(24), fill_value = 0)


plt.figure(figsize=(10,5))
plt.plot(songs_per_hour.index, songs_per_hour.values, marker = 'o', color = 'black')

plt.title(f"Songs Listened to on {selected_date}")
plt.xlabel(f"Time of Day")
plt.ylabel(f"Number of Songs Listened To")

plt.xticks(range(24))
plt.grid(True, linestyle = "--", alpha = 0.6)
plt.tight_layout()
plt.show()    


"""
Comparing Person 1 and Person 2 Top Artists 
"""
# --- Person 1 Top 10 Songs ---
top_10_artist_count = df["Artist_Name"].value_counts()
top_10_artist_count = top_10_artist_count[top_10_artist_count >= 1].head(10)

top_10_artist_df_person1 = pd.DataFrame({"Artist_Name" : top_10_artist_count.index, "Listen Count" : top_10_artist_count.values})
person1_top_artist = top_10_artist_df_person1["Artist_Name"]


# --- Person 2 Top 10 Songs ---
top_10_artist_count_person_2 = df_person_2["Artist_Name"].value_counts()
top_10_artist_count_person_2 = top_10_artist_count_person_2[top_10_artist_count_person_2 >= 1].head(10)

top_10_artist_df_person2 = pd.DataFrame({"Artist_Name" : top_10_artist_count_person_2.index, "Listen Count" : top_10_artist_count_person_2.values})
person2_top_artist = top_10_artist_df_person2["Artist_Name"]


# --- Horizontal Bar Graph With Person 1 Top Artists ---
p1_in_p2_df = df_person_2[df_person_2["Artist_Name"].isin(person1_top_artist)]
p1_in_p2_df_counts = (p1_in_p2_df["Artist_Name"].value_counts().reindex(top_10_artist_count.index).fillna(0).astype(int))


index = np.arange(len(top_10_artist_count.index))
bar_height = 0.35


plt.barh(index - bar_height/2, top_10_artist_count, bar_height, label = 'Person 1', color = '#5F9EA0')
plt.barh(index + bar_height/2, p1_in_p2_df_counts, bar_height, label ='Person 2', color = '#4682B4')

plt.yticks(index, top_10_artist_count.index, fontsize = 8)
plt.xticks(fontsize=8) 

plt.xlabel('Number of Listens', fontsize = 8)
plt.title('Person 1 Top 10 Artists')
plt.legend(fontsize = 9)

for i, (v1, v2) in enumerate(zip(top_10_artist_count.values, p1_in_p2_df_counts.reindex(top_10_artist_count.index, fill_value = 0).values)):
    plt.text(v1 + 0.5, i - bar_height/2, str(v1), va = 'center', fontsize = 7)
    plt.text(v2 + 0.5, i + bar_height/2, str(v2), va = 'center', fontsize = 7)

plt.gca().invert_yaxis() 
plt.tight_layout()
plt.show()



# --- Horizontal Bar Graph With Person 2 Top Artists ---
p2_in_p1_df = df[df["Artist_Name"].isin(person2_top_artist)]
p2_in_p1_df_counts = (p2_in_p1_df["Artist_Name"].value_counts().reindex(top_10_artist_count_person_2.index).fillna(0).astype(int))


index = np.arange(len(top_10_artist_count_person_2.index))
bar_height = 0.35


plt.barh(index - bar_height/2, p2_in_p1_df_counts, bar_height, label = 'Person 1', color = '#5F9EA0')
plt.barh(index + bar_height/2, top_10_artist_count_person_2, bar_height, label ='Person 2', color = '#4682B4')

plt.yticks(index, top_10_artist_count_person_2.index, fontsize = 8)
plt.xticks(fontsize = 8) 

plt.xlabel('Number of Listens', fontsize = 8)
plt.title('Person 2 Top 10 Artists')
plt.legend(fontsize = 9)

for i, (v1, v2) in enumerate(zip(p2_in_p1_df_counts.reindex(top_10_artist_count_person_2.index, fill_value = 0).values, top_10_artist_count_person_2.values)):
    plt.text(v1 + 0.5, i - bar_height/2, str(v1), va = 'center', fontsize = 7)
    plt.text(v2 + 0.5, i + bar_height/2, str(v2), va = 'center', fontsize = 7)

plt.gca().invert_yaxis() 
plt.tight_layout()
plt.show()








































