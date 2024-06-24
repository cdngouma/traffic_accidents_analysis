# US Traffic Accidents Analysis
## 1. Project Overview
## 2. Data Processing
### 2.1. Data Overview
The US Traffic Accident dataset from [Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) provides comprehensive information on traffic accidents across the United States from 2016 to 2023. It contains 7,728,394 rows and 46 columns, each representing a different attribute of the accidents. Here's a brief overview of some of the key columns:

* `ID`: Unique identifier for each accident.
* `Source`: Source of the accident report (e.g., 911 call, news).
* `Severity`: Accident severity rating (typically on a scale from 1 to 4).
* `Start_Time`: Start time of the accident.
* `End_Time`: End time of the accident.
* `Start_Lat/Start_Lng`: Latitude and longitude where the accident started.
* `End_Lat/End_Lng`: Latitude and longitude where the accident ended (many missing values).
* `Distance(mi)`: Distance covered by the accident.
* `Description`: Brief description of the accident.
* `Street`, `City`, `County`, `State`, `Zipcode`: Location details of the accident.
* `Country`: Country where the accident occurred (all should be the USA).
* `Timezone`: Timezone of the accident location.
* `Airport_Code`: Nearest airport to the accident location.
* `Weather_Timestamp`: Time when weather data was recorded.
* `Temperature(F)`, `Wind_Chill(F)`, `Humidity(%)`, `Pressure(in)`, `Visibility(mi)`, `Wind_Direction`, `Wind_Speed(mph)`, `Precipitation(in)`, `Weather_Condition`: Various weather-related attributes.
* `Amenity`, `Bump`, `Crossing`, `Give_Way`, `Junction`, `No_Exit`, `Railway`, `Roundabout`, `Station`, `Stop`, `Traffic_Calming`, `Traffic_Signal`, `Turning_Loop`: Boolean indicators for the presence of specific road features.
* `Sunrise_Sunset`, `Civil_Twilight`, `Nautical_Twilight`, `Astronomical_Twilight`: Time of day indicators related to the position of the sun.

As we want to predict accident severity, some columns may be redundant or not useful for our predictive modeling. These would be `ID`, `Description`, `Source`, `End_Lat/End_Lng`, `Country`, and `Airport_Code`.