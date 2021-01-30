#!/usr/bin/env python
# coding: utf-8

# # Segmenting and Clustering Neighbourhoods in Toronto

# ### Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import requests


# ### Importing Data from Wikipedia

# In[2]:


url = "https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M"
wiki_url = requests.get(url)
wiki_url


# In[3]:


wiki_data = pd.read_html(wiki_url.text)
wiki_data


# In[4]:


len(wiki_data), type(wiki_data)


# In[8]:


wiki_data = wiki_data[0]
wiki_data


# In[9]:


df = wiki_data[wiki_data["Borough"] != "Not assigned"]
df


# In[10]:


df.groupby(['Postal Code']).first()


# In[11]:


len(df['Postal Code'].unique())


# In[12]:


df[df['Borough'] == 'Not assigned']


# In[13]:


df.shape


# # Question 2

# In[14]:


pip install geocoder


# In[15]:


import geocoder


# In[16]:


url = 'http://cocl.us/Geospatial_data'


# In[18]:


df_geo = pd.read_csv(url)
df_geo.head()


# In[20]:


df_geo.dtypes


# In[21]:


df.dtypes


# In[22]:


df.shape


# In[23]:


df_geo.shape


# In[24]:


df = df.join(df_geo.set_index('Postal Code'), on='Postal Code')
df


# In[27]:


df = df.reset_index()


# In[28]:


df.drop(['index'], axis = 'columns', inplace = True)


# In[29]:


df


# # Question 3

# In[30]:


get_ipython().system('conda install -c conda-forge geocoder --yes')
import geocoder
from geopy.geocoders import Nominatim 

address = 'Toronto, Ontario'

geolocator = Nominatim(user_agent="toronto_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Toronto are {}, {}.'.format(latitude, longitude))


# In[40]:


import folium


# In[43]:


#Creating the map of Toronto
map_Toronto = folium.Map(location=[latitude, longitude], zoom_start=11)

# adding markers to map
for latitude, longitude, borough, neighbourhood in zip(df['Latitude'], df['Longitude'], df['Borough'], df['Neighbourhood']):
    label = '{}, {}'.format(neighbourhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [latitude, longitude],
        radius=5,
        popup=label,
        color='red',
        fill=True
        ).add_to(map_Toronto)  
    
map_Toronto


# #### Initializing Foursquare API credentials

# In[44]:


CLIENT_ID = 'JELNUIAY01PO1WFUT31XYY0VR0UVIGKQ1XBJL3HQH45FOZKY' 
CLIENT_SECRET = 'UCSA2RS4N0QT42SGNRTFYYRFDA2AKAJWUAHJYI0XL4HKMNBZ'
VERSION = '20210130' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[46]:



def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius
            )
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighbourhood', 
                  'Neighbourhood Latitude', 
                  'Neighbourhood Longitude', 
                  'Venue', 
                  'Venue Category']
    
    return(nearby_venues)


# In[48]:


venues_in_toronto = getNearbyVenues(df['Neighbourhood'], df['Latitude'], df['Longitude'])


# In[49]:


venues_in_toronto.shape


# In[50]:


venues_in_toronto.head()


# In[51]:


venues_in_toronto.groupby('Neighbourhood').head()


# In[52]:


venues_in_toronto.groupby('Venue Category').max()


# ### One Hot encoding the venue Categories

# In[54]:


toronto_venue_cat = pd.get_dummies(venues_in_toronto[['Venue Category']], prefix="", prefix_sep="")
toronto_venue_cat


# In[55]:


toronto_venue_cat['Neighbourhood'] = venues_in_toronto['Neighbourhood'] 

# moving neighborhood column to the first column
fixed_columns = [toronto_venue_cat.columns[-1]] + list(toronto_venue_cat.columns[:-1])
toronto_venue_cat = toronto_venue_cat[fixed_columns]

toronto_venue_cat.head()


# In[56]:


toronto_grouped = toronto_venue_cat.groupby('Neighbourhood').mean().reset_index()
toronto_grouped.head()


# In[57]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[58]:


import numpy as np


# In[60]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighbourhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighbourhood'] = toronto_grouped['Neighbourhood']

for ind in np.arange(toronto_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(toronto_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# In[61]:


# import k-means from clustering stage
from sklearn.cluster import KMeans


# In[62]:


# set number of clusters
k_num_clusters = 5

toronto_grouped_clustering = toronto_grouped.drop('Neighbourhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=k_num_clusters, random_state=0).fit(toronto_grouped_clustering)
kmeans


# In[63]:


kmeans.labels_[0:100]


# In[64]:


neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)


# In[66]:


toronto_merged = df

toronto_merged = toronto_merged.join(neighborhoods_venues_sorted.set_index('Neighbourhood'), on='Neighbourhood')

toronto_merged.head()


# In[67]:


toronto_merged_nonan = toronto_merged.dropna(subset=['Cluster Labels'])


# In[68]:


import matplotlib.cm as cm
import matplotlib.colors as colors


# In[69]:


map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(k_num_clusters)
ys = [i + x + (i*x)**2 for i in range(k_num_clusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(toronto_merged_nonan['Latitude'], toronto_merged_nonan['Longitude'], toronto_merged_nonan['Neighbourhood'], toronto_merged_nonan['Cluster Labels']):
    label = folium.Popup('Cluster ' + str(int(cluster) +1) + '\n' + str(poi) , parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[int(cluster-1)],
        fill=True,
        fill_color=rainbow[int(cluster-1)]
        ).add_to(map_clusters)
        
map_clusters


# ##### cluster 1

# In[70]:


toronto_merged_nonan.loc[toronto_merged_nonan['Cluster Labels'] == 0, toronto_merged_nonan.columns[[1] + list(range(5, toronto_merged_nonan.shape[1]))]]


# ##### cluster 2

# In[71]:


toronto_merged_nonan.loc[toronto_merged_nonan['Cluster Labels'] == 1, toronto_merged_nonan.columns[[1] + list(range(5, toronto_merged_nonan.shape[1]))]]


# ##### cluster 3

# In[72]:


toronto_merged_nonan.loc[toronto_merged_nonan['Cluster Labels'] == 2, toronto_merged_nonan.columns[[1] + list(range(5, toronto_merged_nonan.shape[1]))]]


# ##### cluster 4

# In[73]:


toronto_merged_nonan.loc[toronto_merged_nonan['Cluster Labels'] == 3, toronto_merged_nonan.columns[[1] + list(range(5, toronto_merged_nonan.shape[1]))]]


# ##### cluster 5

# In[74]:


toronto_merged_nonan.loc[toronto_merged_nonan['Cluster Labels'] == 4, toronto_merged_nonan.columns[[1] + list(range(5, toronto_merged_nonan.shape[1]))]]


# In[ ]:




