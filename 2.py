import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc
from dash import html



csv_data = pd.read_csv('zomato.csv')
excel_data = pd.read_excel('cocn.xlsx')
exchangerate = pd.read_csv('exchange rate.csv')
name=[]
for index,i in enumerate(csv_data["Country Code"]):
    for index1, j in enumerate(excel_data["Country Code"]):
        if(i==j):
            name.append(excel_data["Country"][index1])
            unique_array = list(set(name))
csv_data.insert(3,"country name",name)
unique_array1 = list(set(csv_data["Currency"]))
indrsval=[]

for ind ,s in enumerate(csv_data["Currency"]):
    for ind1 ,h in enumerate(exchangerate["currency"]):
#         print(s.split("(")[0])
#         print(h)
        if(s.split("(")[0]==h):
            csv_data.at[ind,"inr"]=exchangerate["inr"][ind1]
#             print(csv_data["Average Cost for two"][ind])
#             print(exchangerate["inr"][ind1])
#             print(csv_data["Average Cost for two"][ind]*exchangerate["inr"][ind1])
#             indrsval.append(csv_data["Average Cost for two"][ind]*exchangerate["inr"][ind1])


# csv_data.insert(12,"amount in INR",indrsval)
csv_data.head()
#csv_data.to_csv("zomato_with_currency.csv", index=False)
for ins , i in enumerate(csv_data["Average Cost for two"]):
#     print(i)
#     print(csv_data["inr"][ins])
    indrsval.append(i*csv_data["inr"][ins])
#print(indrsval)
csv_data.insert(13,"in rs",indrsval)
csv_data.to_csv("zomato_with_currency.csv", index=False)
print("done")
# fig = go.Figure(data=[go.Bar(x=csv_data['country name'], y=csv_data['inr'])])

# # Set the axis labels and title
# fig.update_layout(xaxis_title='Country', yaxis_title='Exchange Rate (relative to INR)', title='Exchange Rates Relative to the Indian Rupee')

# # Display the chart
# fig.show()

fig = go.Figure()
fig.add_trace(go.Scatter(x=csv_data['country name'], y=csv_data['inr'], mode='lines', name='line'))

# Set the axis labels and title
fig.update_layout(xaxis_title='Country', yaxis_title='Exchange Rate', title='Currency Comparison between India and Other Countries')

# Display the chart
#fig.show()
        
    
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1('Indian Restaurant Dashboard'),
 
    html.Br(),
    dcc.Graph(figure=fig),
    
    dcc.Dropdown(
        id='city-dd',
        options=[{'label': c, 'value': c} for c in csv_data['City'].unique()],
        value='New Delhi'
    ),
    
    
    
    html.H3(id="fc"),
    html.H3(id="cc"),
    html.H3(id="rc"),
    
    
    dcc.Graph(id='chart4'),
    
    dcc.Dropdown(
        id='country-dropdown',
        options=[{'label': c, 'value': c} for c in csv_data['country name'].unique()],
        value='India'
    ),
    
    dcc.Graph(id='chart1'),
    dcc.Graph(id='chart2'),
    dcc.Graph(id='chart3'),
    

    
])

# Define the callbacks
@app.callback(
    [dash.dependencies.Output('chart1', 'figure'),
     dash.dependencies.Output('chart2', 'figure'),
     dash.dependencies.Output('chart3', 'figure')
     ],
    [dash.dependencies.Input('country-dropdown', 'value')],
    #[dash.dependencies.Input('city-dd', 'value')]
)

def update_charts(country):
    filtered_data = csv_data[csv_data['country name'] == country]
   # print(filtered_data)
    # Chart 1: Most expensive cuisines
    expensive_cuisines = filtered_data.groupby('Cuisines')['in rs'].mean().sort_values(ascending=False)[:10]
    fig1 = px.bar(expensive_cuisines, x=expensive_cuisines.index, y='in rs', title='Most Expensive Cuisines')
    
    # Chart 2: Online delivery vs dine-in
    delivery_type_counts = filtered_data['Has Online delivery'].value_counts()
    fig2 = px.pie(delivery_type_counts, values=delivery_type_counts.values, names=delivery_type_counts.index, title='Online Delivery vs Dine-in')
    
    
    # Create a bar chart that shows the comparison between the cities
    
    counts = filtered_data.groupby("City")["Restaurant ID"].count()
    # print(counts)
    fig3 = px.bar(x=counts.index, y=counts.values, labels={"x": "City", "y": "Number of Restaurants"},title='City and Resturant Comparision')

    return fig1,fig2,fig3

@app.callback(
    [dash.dependencies.Output('fc', 'children'),dash.dependencies.Output('cc', 'children'),dash.dependencies.Output('rc', 'children'),dash.dependencies.Output('chart4', 'figure')],
    [dash.dependencies.Input('city-dd', 'value')]
    
)
def update_charts1(city):
    delhi_restaurants = csv_data[csv_data["City"] == city]

# Extract the cuisines for each restaurant in Delhi
    delhi_cuisines = delhi_restaurants["Cuisines"].str.split(", ")

    # Count the occurrence of each cuisine in Delhi
    cuisine_counts = {}
    for cuisines in delhi_cuisines:
        for cuisine in cuisines:
            cuisine = cuisine.strip()
            if cuisine not in cuisine_counts:
                cuisine_counts[cuisine] = 0
            cuisine_counts[cuisine] += 1

    # Sort the cuisines by the occurrence count in descending order
    popular_cuisines = sorted(cuisine_counts.items(), key=lambda x: x[1], reverse=True)
    
    cost_per_cuisine= delhi_restaurants.groupby("Cuisines")["in rs"].mean().reset_index()
    # print(cost_per_cuisine)
    costliest_cuisine=cost_per_cuisine.loc[cost_per_cuisine["in rs"].idxmax(),"Cuisines"]
    
    rating_test= delhi_restaurants["Aggregate rating"].value_counts().reset_index()
    rating_test.columns=['rating test','count']
    df = rating_test.drop(rating_test[rating_test['rating test'] == 0.0].index)
    rating=df.iloc[0]['rating test']
    count=df.iloc[0]['count']
    # print(df.iloc[0]['count'])
    
    delivery_type_counts = delhi_restaurants['Has Online delivery'].value_counts()
    fig3 = px.pie(delivery_type_counts, values=delivery_type_counts.values, names=delivery_type_counts.index, title='Online Delivery vs Dine-in of {}'.format(city))
    # print(costliest_cuisine)    

    # Print the most popular cuisine in Delhi
    return "The famous Cuisines in {} is {}".format(city,popular_cuisines[0][0]), "The costliest cuisine in {} is {}".format(city,costliest_cuisine),  "The rating test of city {} is {} and count is {}".format(city,rating,count),fig3


if __name__ == '__main__':
    app.run_server(debug=True)




          
            