#%%
import streamlit as st
import numpy as np
import pandas as pd
from yahoo_fin import stock_info as si
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime, timedelta

st.set_page_config(page_title='Stock Rotation', layout='wide')

# Layout
left_column, right_column = st.columns([1, 2])

with left_column:
    # Dropdowns
    symbol = st.selectbox('Select Symbol', ['Sectors', 'Stocks', 'Crypto'], index=0)
    time_frame = st.selectbox('Select Time Frame', ['1d', '1h', '1wk'], index=0)

    # Settings
    points = st.selectbox('Points', list(range(1, 20)), index=9)
    rolling_mean1 = st.selectbox('Mean1', list(range(1, 20)), index=6)
    rolling_mean2 = st.selectbox('Mean2', list(range(1, 20)), index=9)

def get_data(IDS, work, time_frame, start_date, end_date):
    out_df = pd.DataFrame()
    for ID in IDS:
        if time_frame == '1d':
            data = si.get_data(ID, start_date=start_date, end_date=end_date, interval='1d')
        elif time_frame == '1h':
            data = si.get_data(ID, start_date=start_date, end_date=end_date, interval='1h')
        elif time_frame == '1wk':
            data = si.get_data(ID, start_date=start_date, end_date=end_date, interval='1wk')
        out_df[ID] = data['close']
    out_df = out_df.replace('NaN', pd.NA).dropna()
    return out_df

def calc_data(out_df, IDS, rolling_mean1, rolling_mean2):
    for ID in IDS:
        out_df[ID + '_avg'] = out_df[ID].ewm(span=rolling_mean1, adjust=False).mean()
    for ID in IDS:
        if ID != 'SPY':
            out_df[ID + '_rel'] = 100 * (out_df[ID + '_avg'] / out_df['SPY_avg'])
            out_df[ID + '_rel_std'] = out_df[ID + '_rel'].std()
            out_df[ID + '_rel_mean'] = out_df[ID + '_rel'].mean()
            out_df[ID + '_RS_Norm'] = 100 + ((out_df[ID + '_rel'] - out_df[ID + '_rel_mean']) / out_df[ID + '_rel_std']) + 1
            out_df[ID + '_rel_ROC'] = out_df[ID + '_rel'].pct_change()
            out_df[ID + '_rel_ROC_std'] = out_df[ID + '_rel_ROC'].std()
            out_df[ID + '_rel_ROC_mean'] = out_df[ID + '_rel_ROC'].mean()
            out_df[ID + '_RS_MOM_Norm'] = 100 + ((out_df[ID + '_rel_ROC'] - out_df[ID + '_rel_ROC_mean']) / out_df[ID + '_rel_ROC_std']) + 1
            out_df[ID + '_RS_Norm'] = out_df[ID + '_RS_Norm'].ewm(span=rolling_mean2, adjust=False).mean()
            out_df[ID + '_RS_MOM_Norm'] = out_df[ID + '_RS_MOM_Norm'].ewm(span=rolling_mean2, adjust=False).mean()
    out_df = out_df.dropna()
    return out_df

def plot_chart(x_data, y_data, labels):
    max_xvalues = max([max(i) for i in x_data])
    max_yvalues = max([max(i) for i in y_data])
    min_xvalues = min([min(i) for i in x_data])
    min_yvalues = min([min(i) for i in y_data])
    axis_padding = 1

    layout = go.Layout(
        xaxis=dict(range=[min_xvalues - axis_padding, max_xvalues + axis_padding], autorange=False, zeroline=True, gridcolor='lightgray',
                   showgrid=True, showline=True, showticklabels=True,
                   tickcolor='black', ticks='outside', tickwidth=1,
                   title='relative strength'),
        yaxis=dict(range=[min_yvalues - axis_padding, max_yvalues + axis_padding], autorange=False, zeroline=True, gridcolor='lightgray',
                   showgrid=True, showline=True, showticklabels=True,
                   tickcolor='black', ticks='outside', tickwidth=1,
                   title='momentum of relative strength'),
        showlegend=False,
        annotations=[
            dict(x=100 - 0.5 * (max_xvalues - min_xvalues), y=100 - 0.5 * (max_yvalues - min_yvalues), text='Lagging', showarrow=False, font=dict(color='red', size=12)),
            dict(x=100 + 0.5 * (max_xvalues - min_xvalues), y=100 - 0.5 * (max_yvalues - min_yvalues), text='Weakening', showarrow=False, font=dict(color='orange', size=12)),
            dict(x=100 + 0.5 * (max_xvalues - min_xvalues), y=100 + 0.5 * (max_yvalues - min_yvalues), text='Leading', showarrow=False, font=dict(color='green', size=12)),
            dict(x=100 - 0.5 * (max_xvalues - min_xvalues), y=100 + 0.5 * (max_yvalues - min_yvalues), text='Improving', showarrow=False, font=dict(color='blue', size=12))
        ]
    )

    shapes = [
        dict(type='rect', x0=50, x1=100, y0=100, y1=150, fillcolor='blue', opacity=0.1, layer='below', line=dict(width=0)),
        dict(type='rect', x0=100, x1=150, y0=100, y1=150, fillcolor='green', opacity=0.1, layer='below', line=dict(width=0)),
        dict(type='rect', x0=100, x1=150, y0=50, y1=100, fillcolor='yellow', opacity=0.1, layer='below', line=dict(width=0)),
        dict(type='rect', x0=50, x1=100, y0=50, y1=100, fillcolor='red', opacity=0.1, layer='below', line=dict(width=0))
    ]

    layout.update(shapes=shapes, autosize=False, width=600, height=600, margin=dict(l=20, r=20, t=20, b=20))

    fig = go.Figure(layout=layout)

    for i in range(len(x_data)):
        fig.add_trace(go.Scatter(x=x_data[i], y=y_data[i], mode='lines', name=labels[i],
                                 line=dict(color=px.colors.qualitative.Plotly[i], width=2), connectgaps=True, line_shape='spline'))
        fig.add_trace(go.Scatter(x=[x_data[i][-1]], y=[y_data[i][-1]], text=labels[i], textposition="top center",
                                 textfont_color=px.colors.qualitative.Plotly[i], marker=dict(color=px.colors.qualitative.Plotly[i], size=10)))

    fig.update_traces(hoverinfo='text+name', mode='lines+markers+text')
    fig.update_layout(xaxis_range=[min_xvalues - axis_padding, max_xvalues + axis_padding], yaxis_range=[min_yvalues - axis_padding, max_yvalues + axis_padding])

    return fig

# Example usage
IDS = ['AAPL', 'MSFT', 'GOOGL', 'SPY']
work = False
time_frame = '1d'
start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
end_date = datetime.now().strftime('%Y-%m-%d')

data = get_data(IDS, work, time_frame, start_date, end_date)
calculated_data = calc_data(data, IDS, rolling_mean1, rolling_mean2)

x_data = [calculated_data[ID + '_RS_Norm'] for ID in IDS if ID != 'SPY']
y_data = [calculated_data[ID + '_RS_MOM_Norm'] for ID in IDS if ID != 'SPY']
labels = [ID for ID in IDS if ID != 'SPY']

with right_column:
    fig = plot_chart(x_data, y_data, labels)
    st.plotly_chart(fig)
