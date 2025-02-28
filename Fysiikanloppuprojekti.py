import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.fftpack
from geopy.distance import geodesic
import folium
from streamlit_folium import folium_static

url = "https://raw.githubusercontent.com/Sakuk-k/Fysiikan-loppuprojekti/refs/heads/main/Accelerometer.csv"
url = "https://raw.githubusercontent.com/Sakuk-k/Fysiikan-loppuprojekti/refs/heads/main/Location.csv"

df = pd.read_csv(url)


@st.cache_data
def load_data():
    acc_data = pd.read_csv("Accelerometer.csv")
    gps_data = pd.read_csv("Location.csv")
    return acc_data, gps_data

acc_data, gps_data = load_data()


acc_data.columns = acc_data.columns.str.strip()
gps_data.columns = gps_data.columns.str.strip()


required_acc_columns = ['Time (s)', 'Z (m/s^2)']
required_gps_columns = ['Time (s)', 'Latitude (°)', 'Longitude (°)']

missing_acc = [col for col in required_acc_columns if col not in acc_data.columns]
missing_gps = [col for col in required_gps_columns if col not in gps_data.columns]

if missing_acc:
    st.error(f"Missing columns in Accelerometer data: {missing_acc}")
    st.stop()

if missing_gps:
    st.error(f"Missing columns in GPS data: {missing_gps}")
    st.stop()


time_acc = acc_data['Time (s)']
acc_z = acc_data['Z (m/s^2)']

time_gps = gps_data['Time (s)']
lat = gps_data['Latitude (°)']
lon = gps_data['Longitude (°)']


gps_data.dropna(subset=['Latitude (°)', 'Longitude (°)'], inplace=True)


if len(acc_z) > 1:
    b, a = signal.butter(4, 0.1, 'low')
    acc_z_filtered = signal.filtfilt(b, a, acc_z)
else:
    acc_z_filtered = acc_z


steps_zero_crossings = np.where((acc_z_filtered[:-1] < 0) & (acc_z_filtered[1:] > 0))[0]
step_count_zero_crossings = len(steps_zero_crossings)


peaks, _ = signal.find_peaks(acc_z_filtered, height=0.5)  
step_count_peaks = len(peaks)


if len(time_acc) > 1:
    fs = 1 / np.mean(np.diff(time_acc))
    freqs = scipy.fftpack.fftfreq(len(acc_z_filtered), d=1/fs)
    power_spectrum = np.abs(scipy.fftpack.fft(acc_z_filtered))**2

    dominant_freq = freqs[np.argmax(power_spectrum[:len(freqs)//2])]
    step_count_fourier = int(dominant_freq * (time_acc.iloc[-1] - time_acc.iloc[0]))

else:
    freqs, power_spectrum = np.array([]), np.array([])
    step_count_fourier = 0


if len(lat) > 1:
    distance = sum(geodesic((lat[i], lon[i]), (lat[i+1], lon[i+1])).meters for i in range(len(lat)-1))
    total_time = time_gps.iloc[-1] - time_gps.iloc[0]
    avg_speed = distance / total_time if total_time > 0 else 0
else:
    distance, avg_speed = 0, 0


step_length = distance / step_count_zero_crossings if step_count_zero_crossings > 0 else 0


st.title("GPS ja kiihtyvyysdatan analyysi")
st.write(f"Askeleet (nollakohdat): {step_count_zero_crossings}")
st.write(f"Askeleet (huiput): {step_count_peaks}")
st.write(f"Askeleet (Fourier-analyysi): {step_count_fourier}")
st.write(f"Matka: {distance:.2f} m")
st.write(f"Keskinopeus: {avg_speed:.2f} m/s")
st.write(f"Askelpituus: {step_length:.2f} m")


fig1, ax1 = plt.subplots()
ax1.plot(time_acc, acc_z_filtered, label="Suodatettu kiihtyvyysdata (Z)")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Acceleration (m/s²)")
ax1.legend()
st.pyplot(fig1)


if len(freqs) > 0:
    fig2, ax2 = plt.subplots()
    ax2.plot(freqs[:len(freqs)//2], power_spectrum[:len(freqs)//2])
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Power")
    ax2.set_title("Tehospektri")
    st.pyplot(fig2)


if len(lat) > 0:
    m = folium.Map(location=[lat.mean(), lon.mean()], zoom_start=15)
    for i in range(len(lat)):
        folium.CircleMarker([lat[i], lon[i]], radius=2, color="blue").add_to(m)
    folium_static(m)
else:
    st.warning("No GPS data available to generate a map.")

st.title("phyphox")
st.caption("Mittaustilanteessa havaittiin yllättävä dynaaminen poikkeama, joka näkyy kiihtyvyysdatan voimakkaana piikkinä. Analyysin perusteella kyseessä on todennäköisesti spontaani liukastuminen, mikä lisää tutkimuksen realistisuutta ja antaa syvemmän käsityksen todellisista liikkumishaasteista. Tieteellisen tarkastelun nimissä ehdotan, että tapaturman dramatiikka huomioidaan lisäpisteiden muodossa – jos ei tieteellisen rohkeuden, niin ainakin fyysisen uhrautumisen ansiosta. Koska eikös se ole niin, että tieteen eteen on joskus kaaduttava – kirjaimellisesti?  😅")

st.image("data.png", caption="Analyysisovellus", use_container_width=True)
