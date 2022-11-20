#importing modules
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
from PIL import Image
import pickle 
import statsmodels.api as sm

#Import data
data = pd.read_csv('student-mat.csv', sep=';')
pd.set_option('display.max_columns',None)
data.head()

#Defining plotly functions
def draw_hist(data):
    fig = px.histogram(data, x='G3')
    return fig
    
def draw_heatmap(data):    
    fig = px.density_heatmap(data, x="age", y="G3", labels={'G3':'Final Grade'}, text_auto=True)
    return fig

def draw_scatter(data):
    fig = px.scatter(data, x="G1", y="G3", color="Medu", size='G3', trendline='ols')
    return fig

def draw_animscatter(data):
    fig = px.scatter(data, x="G1", y="G3", animation_frame="age",
               size="G3", color="Medu",
               log_x=True)
    return fig

def draw_funnel(data):
    fig = px.funnel(data, x='G3', y='Medu')
    return fig

def draw_3dfig(data):
    fig = go.Figure(data=go.Surface(z=data, showscale=True))
    fig.update_layout(
        title='Grades',
        width=400, height=400,
        margin=dict(t=40, r=0, l=20, b=20)
    )
    return fig



#Streamlit section

#Title & Page description
st.markdown('# Student Performance ')

#First Image
image_one = Image.open('Measure.jpg')
st.image(image_one, caption='Using Science to Track Student Behavior')

#Part 1 Description
st.text('To better understand what factors may affect the academic performance of students, \n this page gives multiple types of graphs to visualize the correlation \n between some features from different angles, and then explains it \n from several aspects.  ')

#Filter sidebar based on gender
sex = ['M','F','Population']          
option_two = st.sidebar.selectbox('choose your filter based on gender: ', sex)
if option_two == 'M' or option_two =='F':
    gender = data.loc[data['sex'] == option_two]
else:
    gender = data

#Defining variables as functions' output
hist = draw_hist(gender)
heatmap = draw_heatmap(gender)
scat = draw_scatter(gender)
animscat = draw_animscatter(gender)
funnel = draw_funnel(gender)
fig_3d = draw_3dfig(gender)

#Creating selectbox with multiple plot options
#Adding a description button to each plot
lst = ('No Selection', 'Histogram', 'Heatmap', 'Scatter', 'Animated Scatter', 'Funnel', '3D Figure')
option = st.selectbox('Choose your style: ', lst)

if option == 'Histogram':
    st.plotly_chart(hist, use_container_width=True)
    button = st.button('Description')
    if button:
        st.text('This Histogram describes how the value of the final grade \n is distributed along the population (in this case students), \n we can notice that the population follows Normal distribution.')
if option == 'Scatter':
    st.plotly_chart(scat, use_container_width=True)
    button = st.button('Description')
    if button:
        st.text('Scatter plots gives a detailed visualisation of how the first grades \n can predict the upcoming behavior of the student, \n on the other hand another factor may also interfere in the process: Mother Education. \n When colors are more darker means a higher level of mother educational level. ')
if option == 'Animated Scatter':
    st.plotly_chart(animscat, use_container_width=True)
    button = st.button('Description')
    st.text('The animated scatter takes the same concept of the classical scatter plot, \n but with an additional changing factor, the age. \n By clicking on the play button, you can visualize the transformation of the given population \n in terms of age.')
if option == 'Funnel':
    st.plotly_chart(funnel, use_container_width=True)
    button = st.button('Description')
    if button:
        st.text('Funnel is a type of pipe plot. In this case, \n it simply describes how mother education can be an important factor, \n in maintaining the student performance.')
if option == '3D Figure':
    st.plotly_chart(fig_3d, use_container_width=True)
    button = st.button('Description')
    if button:
        st.text('The entire given informations are visualized in this three dimensional figure.\n It is hard to gain any insights, as it is chaotic and almost randomly distributed.')
if option == 'Heatmap':
    st.plotly_chart(heatmap, use_container_width=True)
    button = st.button('Description')
    if button:
        st.text('This graph is called heatmap. We can see how age can affect the overall performance, \n  as if age increase the number of students having a higher grade decrease.')        

#Second Image
image_two = Image.open('success.jpg')
st.image(image_two, caption='What to do you need to measure success?')

#Part 2 Description
st.text('In order to fill the gaps in student academic performance, several technologies in \n machine learning enter the field of pedagogy. \n One of the method used in this article is a multi class classification techniques \n that helps predict the outcome of the student performance!')
st.text('Three models are trained, and tuned.')

#Loading Pre-Trained Model (Logistic Regression)
model = pickle.load(open('model_one.sav', 'rb'))

#New Data Prediction Part
st.markdown('### Predict Yourself!')
image_one = Image.open('graph.png')

#Third Image
st.image(image_one, caption='Top 10 most affecting features.')
st.text('Based on top 10 features that affects academic performance, \n You can use one of our most successful model, to predict your own performance:')
st.text('Please fill this quick survey and you will get your answer immediately!')

#Survey Section
Medu = st.selectbox('Mother Education Level', ['No Selection','None', 'Primary', 'Secondary', 'Higher'])
Fedu = st.selectbox('Father Education Level', ['No Selection','None', 'Primary', 'Secondary', 'Higher'])
health = st.selectbox('Current Health State', ['No Selection','very bad', 'bad', 'not bad', 'good', 'very good'])
absences = st.number_input('Number of Absences', min_value=0, max_value=93)
age = st.number_input('Your age', min_value=15, max_value=22)
freetime = st.selectbox('Freetime after School', ['No Selection','very low', 'low', 'normal', 'high', 'very high'])
goout = st.selectbox('Going Out with Friends', ['No Selection','very low', 'low', 'normal', 'high', 'very high'])
G1 = st.number_input('Fisrt Period Grade', min_value=0, max_value=20)
G2 = st.number_input('Second Period Grade', min_value=0, max_value=20)
failures = st.number_input('Past Class Failures', min_value=0, max_value=4)

#Mapping characters to numbers
edu = {'None': 0, 'Primary': 1, 'Secondary': 2, 'Higher': 3}
no_study = {'very low': 1, 'low': 2, 'normal': 3, 'high': 4, 'very high': 5}
health_map = {'very bad': 1, 'bad': 2, 'not bad': 3, 'good': 4, 'very good': 5}

#Predict Button
predict = st.button('Predict My Performance')

#Audio Files Loading
audio_file_one = open('Loser.ogg', 'rb')
audio_bytes_one = audio_file_one.read()

audio_file_two = open('Excellent.ogg', 'rb')
audio_bytes_two = audio_file_two.read()

audio_file_three = open('Pass.ogg', 'rb')
audio_bytes_three = audio_file_three.read()


#Predict Part
if predict:
    ph = st.empty()
    sound = st.empty()
#    timer = st.empty()
    original_title = '<p style="font-family:Courier; color:#87CEEB; font-size: 30px;">Are You Ready?</p>'
    ph.markdown(original_title, unsafe_allow_html=True)
#    ph.markdown('## Are You Ready?')
#    N = 3
#    for secs in range(N,0,-1):
#        mm, ss = secs//60, secs%60
#        timer.metric("Countdown", f"{mm:02d}:{ss:02d}")
#        time.sleep(1)
#    timer.empty()
    html_string = """
            <audio controls autoplay>
              <source src="https://storage.cloudconvert.com/tasks/340e37f9-68b3-4318-a75e-5d9fb275c0e9/mixkit-drum-roll-566.mp3?AWSAccessKeyId=cloudconvert-production&Expires=1669028073&Signature=ZLZPSMpARklvHGlbtrFGa638cPw%3D&response-content-disposition=attachment%3B%20filename%3D%22mixkit-drum-roll-566.mp3%22&response-content-type=audio%2Fmpeg" type="audio/mp3">
            </audio>
            """
    sound.markdown(html_string, unsafe_allow_html=True)
    time.sleep(4.5)  
    sound.empty()  
    ph.empty()
    
    pred = model.predict(np.array([edu[Medu], edu[Fedu], health_map[health], absences, age, no_study[goout], no_study[freetime], G1, G2, failures]).reshape(1,-1))    
    if pred == 0:
        st.audio(audio_bytes_one, format='audio/ogg')
        st.write('### You will fail! 😿')
        #st.snow()
    elif pred == 1:
        st.audio(audio_bytes_three, format='audio/ogg')
        st.write('### You will Pass! 🥲')
    else:
        st.audio(audio_bytes_two, format='audio/ogg')
        st.write('### Excellent Outcome! 🥳')
        time.sleep(1)
        st.balloons()
        
#Fourth Image
image_three = Image.open('predict.png')
st.image(image_three, caption='AI helps improving performance!')
st.markdown('## Thanks For Your Visit!')
    
#end of the project
#thank you
