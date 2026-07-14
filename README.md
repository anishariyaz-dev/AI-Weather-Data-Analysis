#  Predictive Weather Analytics Platform

An advanced machine learning and glassmorphic data pipeline framework built with Streamlit. This platform bypasses traditional, rigid UI templates to offer a frosted pastel glassmorphic design while delivering real-time atmospheric diagnostics and predictive temperature forecasting powered by AI.


 Key Features

* Premium Glassmorphic UI: Custom-injected CSS providing a responsive, modern layout utilizing the *Plus Jakarta Sans* typeface.
* No-Sidebar Interface: Bypasses conventional Streamlit structure to deliver a streamlined "Control Tower" dashboard at the top of the canvas.
* Geocoding Core: Employs the Open-Meteo Geocoding API to seamlessly resolve arbitrary global city names into precise geographic coordinates.
* Smart Server-Side Caching: Utilizes time-to-live (TTL) bounded data caching strategies to minimize API traffic and maximize speed.
* Autoregressive ML Forecaster: Implements a dynamic machine learning pipeline that uses its own subsequent predictions to project temperature trends up to 7 days into the future.
* Plotly Data Visualization: Generates smooth vector-rendered time-series line plots with clean hover interactions.


 Technical Breakdown

 1. UI/UX Engineering
The interface overrides the virtual DOM elements of Streamlit through raw HTML injection to enforce the glassmorphic styling:
* Typography:** Integrates the *Plus Jakarta Sans* font family globally to guarantee smooth kerning.
* Glassmorphic Elements: Implements a distinct CSS template class (`.glass-card`) configured with a blur backdrop-filter and semi-transparent borders.
* High-Contrast Input Fields: Targets underlying input components directly to force white input states and clear typography, completely preventing contrast bugs during system dark-mode switches.

 2. Data Layer & API Architecture
The application operates on a 3-tier API consumption paradigm utilizing Python's `requests` package:

Geocoding- `geocoding-api.open-meteo.com` 
Diagnostics- `api.open-meteo.com` 
Archive-`archive-api.open-meteo.com` 

 3. Machine Learning Engine
The mathematical predictive baseline utilizes a specialized `RandomForestRegressor` initialized with 50 micro-decision estimators.

* Month of Observation: Established to capture macro seasonal variance.
* Day of Year: Established to track granular orbital climate tracking.
* Lagged Baseline Target: The maximum temperature recorded on the previous day to capture immediate atmospheric inertia.

To project $N$ days into the future, the model executes an autoregressive sequence where each prediction feeds back into the model as the input for the next day:

$$T_{t+i} = f(\text{month}_{t+i}, \, \text{day\_of\_year}_{t+i}, \, T_{t+i-1})$$


 Installation & Local Deployment
 1. Repository Setup
Clone the project to your local machine:
```bash
git clone - https://github.com/anishariyaz-dev/AI-Weather-Data-Analysis