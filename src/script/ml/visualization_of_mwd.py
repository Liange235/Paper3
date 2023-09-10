from numpy.__config__ import show
import plotly.graph_objects as go
import numpy as np
import os

path = os.getcwd()
# chain = np.loadtxt(path+r"\Data\chain.txt")
# Gp_out_mean = np.loadtxt(path+r"\Data\Y_hat.txt")
# quan05_value = np.loadtxt(path+r"\Data\Y_hat05.txt")
# quan95_value = np.loadtxt(path+r"\Data\Y_hat95.txt")
# real_value = np.loadtxt(path+r"\Data\ValidationY.txt")  
def plot_mwd_animation(x, Gp_out_mean, real_value, quan05_value=None, quan95_value=None, str_n=None):
    n = real_value.shape[0]
    m = real_value.shape[1]
    fd = "MWD"
    st = "MWD"
    str_x = "Chain Length"
    str_t = "Molecular Weight Distribution"
    if m==400:
        fd = "CSD"
        st = "Volumn pdf(mm^-1)"
        str_x = "Particle size(mm)"
        str_t = "Crystal Size Distribution"
    ss_single = np.linalg.norm(real_value-Gp_out_mean, axis=1)
    rmse = np.sqrt(np.mean(ss_single))
    fig_dict = {
        "data": [],
        "layout": {},
        "frames": []
    }
    fig_dict["layout"]["xaxis"] = {"range": [np.min(x), np.max(x)], "title": str_x}
    fig_dict["layout"]["yaxis"] = {"range": [np.min(real_value), np.max(real_value)], "title": st}
    fig_dict["layout"]["plot_bgcolor"] = "rgba(0,0,0,0)"
    fig_dict["layout"]["hovermode"] = "x unified"
    fig_dict["layout"]["title"] = str_n+" Result Of "+str_t+" | RMSE:" + str(round(rmse, 4))
    fig_dict["layout"]["updatemenus"] = [{
            "bordercolor": "rgb(205,133,63)",
            "borderwidth": 4,
            "buttons":[
                {
                    "args": [None, {"frame": {"duration": 1000, "redraw": False},
                                    "fromcurrent": True, "transition": {"duration": 400,
                                                                        "easing": "quadratic-in-out"}}],
                    "label": "Click",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}}],
                    "label": "Stop",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top" 
    }]
    Ob_seq = 1
    data_dict1 = {
        "x": x.T,
        "y": np.round(Gp_out_mean[0,:], 3),
        # "hovertemplate": "Ob_seq: str(%{Ob_seq}) <br>",
        "mode": "lines",
        "legendgroup": "first",
        "text": "Ob_seq: "+str(Ob_seq),
        "line_width": 4, 
        "line_shape": "spline",
        "line_smoothing": 1.3,
        "line_color": "#990099",
        "legendrank": 4,
        "name": str_n,
    }
    fig_dict["data"].append(data_dict1)
    data_dict2 = data_dict1.copy()
    data_dict2['y'] = np.round(real_value[0,:], 3)
    data_dict2['text'] = f"Current_SE: {ss_single[0]:.4f}"
    data_dict2['mode'] = "lines+markers"
    data_dict2['marker_symbol'] = 'triangle-left'
    data_dict2['marker_line_width'] = 4
    data_dict2['line_color'] = "#FF0000"
    data_dict2['name'] = "ObserV"
    data_dict2['legendrank'] = 3
    fig_dict["data"].append(data_dict2)
    data_dict11 = data_dict1.copy()
    data_dict12 = data_dict2.copy()
    if quan05_value is not None:
        data_dict3 = data_dict1.copy()
        data_dict3['y'] = np.round(quan05_value[0,:], 3)
        del data_dict3['text']
        data_dict3['line_color'] = '#00CC00'
        data_dict3['line_dash'] = 'dashdot'
        data_dict3['name'] = "5%_region"
        data_dict3['legendrank'] = 2
        fig_dict["data"].append(data_dict3)
        data_dict4 = data_dict3.copy()
        data_dict4['y'] = np.round(quan95_value[0,:], 3)
        data_dict4['fill'] = 'tonexty'
        data_dict4['name'] = "95%_region"
        data_dict4['legendrank'] = 1
        fig_dict["data"].append(data_dict4)
        data_dict13 = data_dict3.copy()
        data_dict14 = data_dict4.copy()
        for _ in range(n-1):
            frame = {"data": [], "name": str(_)}
            data_dict11['y'] = np.round(Gp_out_mean[_,:], 3)
            data_dict11['text'] = "Ob_seq: "+str(_+1)
            frame["data"].append(data_dict11.copy())
            data_dict12['y'] = np.round(real_value[_,:], 3)
            data_dict12['text'] = f"Current_SE: {ss_single[_+1]:.4f}"
            frame["data"].append(data_dict12.copy())
            data_dict13['y'] = np.round(quan05_value[_,:], 3)
            frame["data"].append(data_dict13.copy())
            data_dict14['y'] = np.round(quan95_value[_,:], 3)
            frame["data"].append(data_dict14.copy())
            fig_dict["frames"].append(frame)
    else:
        for _ in range(n-1):
            frame = {"data": [], "name": str(_)}
            data_dict11['y'] = np.round(Gp_out_mean[_,:], 3)
            data_dict11['text'] = "Ob_seq: "+str(_+1)
            frame["data"].append(data_dict11.copy())
            data_dict12['y'] = np.round(real_value[_,:], 3)
            data_dict12['text'] = f"Current_SE: {ss_single[_+1]:.4f}"
            frame["data"].append(data_dict12.copy())
            fig_dict["frames"].append(frame)
    fig = go.Figure(fig_dict)
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_layout(
        # font_family="Courier New",
        font_color="blue",
        title_font_family="Times New Roman",
        title_font_color="red",
        title_font_size=32,
        legend_title_font_color="green"
    )
    fig.update_xaxes(title_font_family="Times New Roman")
    fig.update_xaxes(title_font_size=32)
    fig.update_yaxes(title_font_family="Times New Roman")
    fig.update_yaxes(title_font_size=32)
    # fig.update_layout(template='plotly_white')
    # fig.show()
    dir = os.path.join(path,'Data','result_set',fd,str_n+'.html')
    fig.write_html(dir, auto_play=False)
    # fig.write_image(path+'\r1.png')
    return None
