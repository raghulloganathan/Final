import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
import os
import matplotlib.pyplot as plt
import numpy as np
from streamlit_option_menu import option_menu

with st.sidebar:
    select = option_menu("Menu", ["Charts","HeatMap & Model"], 
                
                menu_icon= "menu-button-wide",
                default_index=0,
                styles={"nav-link": {"font-size": "24px", "text-align": "left", "margin": "-2px", "--hover-color": "#6F36AD"},
                        "nav-link-selected": {"background-color": "#6F36AD"}})

if select == "Charts":

    url = 'https://raw.githubusercontent.com/nethajinirmal13/Training-datasets/main/Vaccine.csv'
    df = pd.read_csv(url)

    df.h1n1_awareness.fillna(df.h1n1_awareness.median(), inplace=True)
    df.h1n1_worry.fillna(df.h1n1_worry.median(), inplace=True)
    df.antiviral_medication.fillna(df.antiviral_medication.median(), inplace=True)
    df.contact_avoidance.fillna(df.contact_avoidance.median(), inplace=True)
    df.bought_face_mask.fillna(df.bought_face_mask.median(), inplace=True)
    df.wash_hands_frequently.fillna(df.wash_hands_frequently.median(), inplace=True)
    df.avoid_large_gatherings.fillna(df.avoid_large_gatherings.median(), inplace=True)
    df.reduced_outside_home_cont.fillna(df.reduced_outside_home_cont.median(), inplace=True)
    df.avoid_touch_face.fillna(df.avoid_touch_face.median(), inplace=True)
    df.dr_recc_h1n1_vacc.fillna(df.dr_recc_h1n1_vacc.median(), inplace=True)
    df.dr_recc_seasonal_vacc.fillna(df.dr_recc_seasonal_vacc.median(), inplace=True)
    df.chronic_medic_condition.fillna(df.chronic_medic_condition.median(), inplace=True)
    df.cont_child_undr_6_mnths.fillna(df.cont_child_undr_6_mnths.median(), inplace=True)
    df.is_health_worker.fillna(df.is_health_worker.median(), inplace=True)
    df.has_health_insur.fillna(df.has_health_insur.median(), inplace=True)
    df.is_h1n1_vacc_effective.fillna(df.is_h1n1_vacc_effective.median(), inplace=True)
    df.is_h1n1_risky.fillna(df.is_h1n1_risky.median(), inplace=True)
    df.sick_from_h1n1_vacc.fillna(df.sick_from_h1n1_vacc.median(), inplace=True)
    df.is_seas_vacc_effective.fillna(df.is_seas_vacc_effective.median(), inplace=True)
    df.is_seas_risky.fillna(df.is_seas_risky.median(), inplace=True)
    df.sick_from_seas_vacc.fillna(df.sick_from_seas_vacc.median(), inplace=True)
    df.no_of_adults.fillna(df.no_of_adults.median(), inplace=True)
    df.no_of_children.fillna(df.no_of_children.median(), inplace=True)

    string_columns = df.select_dtypes(include=['object']).columns
    df[string_columns] = df[string_columns].fillna(df[string_columns].mode().iloc[0])

    def question1():
        
        # Assuming 'sick_from_seas_vacc' is the correct column name in your DataFrame
        plt.figure(figsize=(10, 8))
        sns.countplot(data=df, x='sick_from_seas_vacc', palette="cubehelix")
        plt.xlabel('sick from seasonal vaccine')
        plt.title('Count of People Sick from Seasonal Vaccine')
        plt.xticks(rotation=45)  # Rotate x-axis labels if needed
        plt.tight_layout()  # Adjust layout to prevent labels from overlapping
        fig = plt.gcf()  # Get the current figure
        st.pyplot(fig)  # Display the plot in Streamlit

    def question2():
        
        # Assuming 'no_of_adults' and 'no_of_children' are columns in your DataFrame df11_
        plt.figure(figsize=(10, 8))
        sns.barplot(data=df, x="no_of_adults", y="no_of_children", linewidth=2, edgecolor='red')
        plt.xlabel("Total No of Adults")
        plt.ylabel("Total No of Children")
        plt.title("Total Number of Children vs. Total Number of Adults")
        plt.tight_layout()  # Adjust layout to prevent labels from overlapping

        # Get the current figure
        fig = plt.gcf()

        # Display the plot in Streamlit
        st.pyplot(fig)

    def question3():
        # Assuming 'has_health_insur' and 'is_health_worker' are columns in your DataFrame df11_
        plt.figure(figsize=(10, 8))
        sns.lineplot(data=df, x="has_health_insur", y="is_health_worker")
        plt.xlabel("Health Insurance")
        plt.ylabel("Health Worker")
        plt.title("Relationship between Health Insurance and Health Worker")
        plt.tight_layout()  # Adjust layout to prevent labels from overlapping
        # Get the current figure
        fig = plt.gcf()
        # Display the plot in Streamlit
        st.pyplot(fig)


    def question4():    
        # Assuming df contains your DataFrame with a column named 'sex'
        # Define custom colors for the pie chart
        colors = ["pink", "lightblue"]  # You can adjust the colors as needed
        # Create the pie chart with custom colors
        fig = px.pie(df, names="sex", width=600, height=500, hole=0.5, color_discrete_sequence=colors)
        # Set the title for the plot
        fig.update_layout(title="Distribution of Sex")
        fig = plt.gcf()
        # Display the plot in Streamlit
        st.plotly_chart(fig)

    def question5():
        # Assuming 'dr_recc_h1n1_vacc' and 'dr_recc_seasonal_vacc' are columns in your DataFrame df11_
        plt.figure(figsize=(10, 8))
        sns.catplot(y="dr_recc_h1n1_vacc", x="dr_recc_seasonal_vacc", height=5, kind="bar", data=df)
        plt.xlabel("Recommended Seasonal Vaccination")
        plt.ylabel("Recommended H1N1 Vaccination")
        plt.title("Recommended Vaccinations by Healthcare Providers")
        plt.tight_layout()  # Adjust layout to prevent labels from overlapping
        # Get the current figure
        fig = plt.gcf()
        # Display the plot in Streamlit with the given title
        st.pyplot(fig)

    def question6():
        # Assuming 'income_level' is the correct column name in your DataFrame df11_
        fig = px.pie(df, names="income_level", width=600, height=500, hole=0.5, color_discrete_sequence=px.colors.sequential.Mint_r)

        # Set the title for the plot
        fig.update_layout(title="Distribution of Income Levels")
        fig = plt.gcf()
        # Display the plot in Streamlit with the given title
        st.plotly_chart(fig)
        


    def question7():
        plt.figure(figsize=(10, 8))
        sns.barplot(data=df, x="housing_status", y="income_level", order=df["housing_status"].unique()[::-1], linewidth=2, edgecolor='red')

        # Set the title for the plot
        plt.title("Income Levels by Housing Status")
        fig = plt.gcf()
        # Display the plot in Streamlit with the given title
        st.pyplot()




    def question8():
        # Assuming correct column name in your DataFrame df1_
        fig = plt.figure(figsize=(10, 5))
        plt.bar(df['is_seas_vacc_effective'], df['is_h1n1_vacc_effective'], color='maroon', width=0.4)
        plt.xlabel('Is Seasonal Vaccine Effective')
        plt.ylabel('Is H1N1 Vaccine Effective')
        plt.title('Effectiveness of H1N1 Vaccine vs Seasonal Vaccine')
        fig = plt.gcf()
        st.pyplot(fig)
        

    ques= st.selectbox("**Select the Options**",('Count of People Sick from Seasonal Vaccine',
                                                'Total Number of Children vs. Total Number of Adults',
                                                'Relationship between Health Insurance and Health Worker',
                                                'Distribution of Sex',
                                                'Recommended Vaccinations by Healthcare Providers',
                                                'Distribution of Income Levels',
                                                'Income Level by Housing Status',
                                                'Effectiveness of H1N1 Vaccine vs Seasonal Vaccine',
                                                'Top 50 Districts With Lowest Transaction Amount'))

    if ques=="Count of People Sick from Seasonal Vaccine":
            question1()

    elif ques=="Total Number of Children vs. Total Number of Adults":
            question2()
            
    elif ques=="Relationship between Health Insurance and Health Worker":
        question3()
            
    elif ques=="Distribution of Sex":
        question4()
        
    elif ques=="Recommended Vaccinations by Healthcare Providers":
        question5()
        
    elif ques=="Distribution of Income Levels":
        question6()
        
    elif ques=="Income Level by Housing Status":
        question7()
        
    elif ques=="Effectiveness of H1N1 Vaccine vs Seasonal Vaccine":
        question8()
        
 

    #types arae charts 

    #______________________________________Bar Plot

    def ty1():#type
        plt.figure(figsize=(10, 6))
        df['sex'].value_counts().plot(kind='bar', color='skyblue')
        plt.xlabel('Sex')
        plt.ylabel('Count')
        plt.title('Distribution of Sex')
        fig = plt.gcf()
        st.pyplot(fig)


    #______________________Histogram:


    def ty2():
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='age_bracket', kde=True)
        plt.xlabel('Age Bracket')
        plt.ylabel('Frequency')
        plt.title('Distribution of Age Bracket')
        fig = plt.gcf()
        st.pyplot(fig)


    #_______________Scatter Plot:
    def ty4():
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='no_of_adults', y='no_of_children', hue='h1n1_vaccine')
        plt.xlabel('Number of Adults')
        plt.ylabel('Number of Children')
        plt.title('Relationship between Number of Adults and Children with H1N1 Vaccine')
        fig = plt.gcf()
        st.pyplot(fig)


    #_____________________Line Plot:
    def ty5():
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='income_level', y='h1n1_awareness')
        plt.xlabel('Income Level')
        plt.ylabel('H1N1 Awareness')
        plt.title('H1N1 Awareness across Income Levels')
        fig = plt.gcf()
        st.pyplot(fig)

    #__________________Box Plot:
    def ty6():
        
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df, x='census_msa', y='h1n1_worry')
            plt.xlabel('Census MSA')
            plt.ylabel('H1N1 Worry')
            plt.title('H1N1 Worry across Census MSA')
            fig = plt.gcf()
            st.pyplot(fig)


    #Count Plot:
    def ty7():
        
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='employment', hue='h1n1_vaccine')
        plt.xlabel('Employment Status')
        plt.ylabel('Count')
        plt.title('Distribution of H1N1 Vaccine across Employment Status')
        fig = plt.gcf()
        st.pyplot(fig)
        
    qa= st.selectbox("**Select the Options**",('Distribution of Sex',
                                                'Distribution of Age Bracket',
                                                'Relationship between Number of Adults and Children with H1N1 Vaccine',
                                                'H1N1 Awareness across Income Levels',
                                                'H1N1 Worry across Census MSA',
                                                'Distribution of H1N1 Vaccine across Employment Status'))

    if qa=="Distribution of Sex":
        ty1()

    elif qa=="Distribution of Age Bracket":
        ty2()
            
    elif qa=="Relationship between Number of Adults and Children with H1N1 Vaccine":
        ty4()
        
    elif qa=="H1N1 Awareness across Income Levels":
        ty5()

    elif qa=="H1N1 Worry across Census MSA":
        ty6()

    elif qa=="Distribution of H1N1 Vaccine across Employment Status":
        ty7()
        

    def sam1():

        # Bar Chart
        plt.figure(figsize=(10, 6))
        plt.bar(df['sex'], df['no_of_children'], color='skyblue')
        plt.xlabel('Sex')
        plt.ylabel('Number of Children')
        plt.title('Number of Children by Sex')
        fig = plt.gcf()
        st.pyplot(fig)  # Display the bar chart in Streamlit
    def sam2():
        # Histogram
        plt.figure(figsize=(10, 6))
        plt.hist(df['age_bracket'], bins=10, color='salmon', edgecolor='black')
        plt.xlabel('Age Bracket')
        plt.ylabel('Frequency')
        plt.title('Distribution of Age Bracket')
        fig = plt.gcf()
        st.pyplot(fig)  # Display the histogram in Streamlit
    def sam3():
        # Pie Chart
        plt.figure(figsize=(8, 8))
        plt.pie(df['race'].value_counts(), labels=df['race'].value_counts().index, autopct='%1.1f%%', startangle=140)
        plt.title('Race Distribution')
        fig = plt.gcf()
        st.pyplot(fig)  # Display the pie chart in Streamlit
    def sam4():
        # Line Plot
        plt.figure(figsize=(10, 6))
        plt.plot(df['income_level'], df['no_of_adults'], marker='o', linestyle='-', color='green')
        plt.xlabel('Income Level')
        plt.ylabel('Number of Adults')
        plt.title('Number of Adults by Income Level')
        plt.xticks(rotation=45)
        plt.grid(True)
        fig = plt.gcf()
        st.pyplot(fig)  # Display the line plot in Streamlit
    def sam5():
        # Box Plot
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='qualification', y='no_of_children', data=df, palette='Set2')
        plt.xlabel('Qualification')
        plt.ylabel('Number of Children')
        plt.title('Number of Children by Qualification')
        plt.xticks(rotation=45)
        fig = plt.gcf()
        st.pyplot(fig)  # Display the box plot in Streamlit


    qa11= st.selectbox("**Select the Option**",('Number of Children by Sex',
                                                'Distribution of Age Bracket',
                                                'Race Distribution',
                                                'Number of Adults by Income Level',
                                                'Number of Children by Qualification'
                                                ))

    if qa11=="Number of Children by Sex":
        sam1()

    if qa11=="Distribution of Age Bracket":
        sam2()

    if qa11=="Race Distribution":
        sam3()

    if qa11=="Number of Adults by Income Level":
        sam4()

    if qa11=="Number of Children by Qualification":
        sam5()


if select=="HeatMap & Model":
    
    import streamlit as st
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    import streamlit as st
    import pandas as pd
    import seaborn as sns
    import plotly.express as px
    import os
    import matplotlib.pyplot as plt
    import numpy as np


    # Assuming you have defined 'x', 'y', 'xtrain', 'xtest', 'ytrain', 'ytest', and 'model' already

    # Display the logistic regression model information in Streamlit
    st.title('Logistic Regression Model')

    url = 'https://raw.githubusercontent.com/nethajinirmal13/Training-datasets/main/Vaccine.csv'
    df11_ = pd.read_csv(url)


    df11_.h1n1_awareness.fillna(df11_.h1n1_awareness.median(), inplace=True)
    df11_.h1n1_worry.fillna(df11_.h1n1_worry.median(), inplace=True)
    df11_.antiviral_medication.fillna(df11_.antiviral_medication.median(), inplace=True)
    df11_.contact_avoidance.fillna(df11_.contact_avoidance.median(), inplace=True)
    df11_.bought_face_mask.fillna(df11_.bought_face_mask.median(), inplace=True)
    df11_.wash_hands_frequently.fillna(df11_.wash_hands_frequently.median(), inplace=True)
    df11_.avoid_large_gatherings.fillna(df11_.avoid_large_gatherings.median(), inplace=True)
    df11_.reduced_outside_home_cont.fillna(df11_.reduced_outside_home_cont.median(), inplace=True)
    df11_.avoid_touch_face.fillna(df11_.avoid_touch_face.median(), inplace=True)
    df11_.dr_recc_h1n1_vacc.fillna(df11_.dr_recc_h1n1_vacc.median(), inplace=True)
    df11_.dr_recc_seasonal_vacc.fillna(df11_.dr_recc_seasonal_vacc.median(), inplace=True)
    df11_.chronic_medic_condition.fillna(df11_.chronic_medic_condition.median(), inplace=True)
    df11_.cont_child_undr_6_mnths.fillna(df11_.cont_child_undr_6_mnths.median(), inplace=True)
    df11_.is_health_worker.fillna(df11_.is_health_worker.median(), inplace=True)
    df11_.has_health_insur.fillna(df11_.has_health_insur.median(), inplace=True)
    df11_.is_h1n1_vacc_effective.fillna(df11_.is_h1n1_vacc_effective.median(), inplace=True)
    df11_.is_h1n1_risky.fillna(df11_.is_h1n1_risky.median(), inplace=True)
    df11_.sick_from_h1n1_vacc.fillna(df11_.sick_from_h1n1_vacc.median(), inplace=True)
    df11_.is_seas_vacc_effective.fillna(df11_.is_seas_vacc_effective.median(), inplace=True)
    df11_.is_seas_risky.fillna(df11_.is_seas_risky.median(), inplace=True)
    df11_.sick_from_seas_vacc.fillna(df11_.sick_from_seas_vacc.median(), inplace=True)
    df11_.no_of_adults.fillna(df11_.no_of_adults.median(), inplace=True)
    df11_.no_of_children.fillna(df11_.no_of_children.median(), inplace=True)

    string_columns = df11_.select_dtypes(include=['object']).columns
    df11_[string_columns] = df11_[string_columns].fillna(df11_[string_columns].mode().iloc[0])

    df11_=df11_.drop(columns=['age_bracket', 'qualification',
        'race', 'sex', 'income_level', 'marital_status', 'housing_status',
        'employment', 'census_msa'])

    # Split the data into features (x) and target (y)
    x = np.array(df11_[['h1n1_worry', 'h1n1_awareness', 'antiviral_medication',
        'contact_avoidance', 'bought_face_mask', 'wash_hands_frequently',
        'avoid_large_gatherings', 'reduced_outside_home_cont',
        'avoid_touch_face', 'dr_recc_h1n1_vacc', 'dr_recc_seasonal_vacc',
        'chronic_medic_condition', 'cont_child_undr_6_mnths',
        'is_health_worker', 'has_health_insur', 'is_h1n1_vacc_effective',
        'is_h1n1_risky', 'sick_from_h1n1_vacc', 'is_seas_vacc_effective',
        'is_seas_risky', 'sick_from_seas_vacc', 
            'no_of_adults', 'no_of_children']])
    y = np.array(df11_[["h1n1_vaccine"]])

    # Split the data into training and testing sets
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.40, random_state=42)

    # Train the logistic regression classifier
    model = LogisticRegression()
    model.fit(xtrain, ytrain)

    # Display the size of training and testing data
    st.write('Training data size:', xtrain.shape)
    st.write('Testing data size:', xtest.shape)

    # Train the logistic regression classifier
    st.write('Training the logistic regression classifier...')

    # Display the training progress if necessary
    with st.spinner('Training in progress...'):
        model = LogisticRegression()
        model.fit(xtrain, ytrain)

    st.write('Training completed!')

    # Display the model parameters
    st.subheader('Model Parameters')
    st.write('Intercept:', model.intercept_)
    st.write('Coefficients:', model.coef_)

    # Displaying the heatmap for the coefficients
    plt.figure(figsize=(12, 8))
    sns.heatmap(model.coef_, annot=True, cmap='coolwarm', xticklabels=df11_.columns[:-1], yticklabels=['Coefficient'])
    plt.title('Heatmap for Model Coefficients')
    plt.xlabel('Features')
    plt.ylabel('Coefficient')
    coeff_heatmap_fig = plt.gcf()  # Get the current figure
    st.pyplot(coeff_heatmap_fig)

    # Displaying the heatmap for the correlation
    plt.figure(figsize=(12, 8))
    sns.heatmap(df11_.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    corr_heatmap_fig = plt.gcf()  # Get the current figure
    st.pyplot(corr_heatmap_fig)
    # Evaluate the model performance
    st.subheader('Model Evaluation')
    train_score = model.score(xtrain, ytrain)
    test_score = model.score(xtest, ytest)
    
    st.write('Training Accuracy:', train_score)
    st.write('Testing Accuracy:', test_score)


    # Display user input fields
    st.title('Vaccine Usage Prediction')
    st.write("Please enter the following information:")

    a = st.slider("H1N1 worry", min_value=0, max_value=3)
    b = st.slider("H1N1 awareness", min_value=0, max_value=2)
    c = st.slider("Antiviral medication", min_value=0, max_value=1)
    d = st.slider("Contact avoidance", min_value=0, max_value=1)
    e = st.slider("Bought face mask", min_value=0, max_value=1)
    f = st.slider("Wash hands frequently", min_value=0, max_value=1)
    g = st.slider("Avoid large gatherings", min_value=0, max_value=1)
    h = st.slider("Reduced outside home cont", min_value=0, max_value=1)
    i = st.slider("Avoid touch face", min_value=0, max_value=1)
    j = st.slider("Dr recc H1N1 vacc", min_value=0, max_value=1)
    k = st.slider("Dr recc seasonal vacc", min_value=0, max_value=1)
    l = st.slider("Chronic medic condition", min_value=0, max_value=1)
    m = st.slider("Cont child undr 6 mnths", min_value=0, max_value=1)
    n = st.slider("Is health worker", min_value=0, max_value=1)
    o = st.slider("Has health insur", min_value=0, max_value=1)
    p = st.slider("Is H1N1 vacc effective", min_value=0, max_value=5)
    q = st.slider("Is H1N1 risky", min_value=0, max_value=5)
    r = st.slider("Sick from H1N1 vacc", min_value=0, max_value=5)
    s = st.slider("Is seas vacc effective", min_value=0, max_value=5)
    t = st.slider("Sick from seas vacc", min_value=0, max_value=5)
    u = st.slider("No of adults", min_value=0, max_value=5)
    v = st.slider("No of children", min_value=0, max_value=3)
    w = st.slider("Is H1N1 risky", min_value=0, max_value=3, key="is_h1n1_risky")

    # Create feature array
    features = np.array([[a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w]])

    # Display predicted vaccine usage
    if st.button('Predict Vaccine Usage'):
        prediction = model.predict(features)
        st.write("Predicted Vaccine Usage:", prediction)
