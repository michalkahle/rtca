from __future__ import unicode_literals

import copy, math, time, json

import numpy
from scipy import signal
from sklearn import metrics
from lmfit import minimize, Parameters, Minimizer



curve_fncs = {
    "3pl": lambda x, bottom, top, inflection: top+(bottom-top)/(1+10**((inflection-x))),
    "4pl": lambda x, bottom, top, inflection, slope: ((bottom-top)/(1+((x/inflection)**slope))) + top,
    "5pl": lambda x, bottom, top, inflection, slope, symmetry: bottom + (top/(1+(x/inflection)**slope)**symmetry),
}

    # bottom = minimum asymptote in an assay where you have a standard curve, this can be thought of as the response value at 0 standard concentration.
    # slope = The Hill Slope or slope factor refers to the steepness of the curve. It could either be positive or negative. As the absolute value of the Hill slope increases, so does the steepness of the curve.
    # inflection = The inflection point is defined as the point on the curve where the curvature changes direction or signs. This can be better explained if you can imagine the concavity of a sigmoidal curve. The inflection point is where the curve changes from being concave upwards to concave downwards (see picture below).
    # top = maximum asymptote in an assay where you have a standard curve, this can be thought of as the response value for infinite standard concentration.
    # symmetry = symmetry is the asymmetry factor

def detect_toxic_concentrations(data_points):
    # for i, values in enumerate(data_points):
    conc2values = {}

    for v in data_points:
        x, y = v[0], v[1]
        if x in conc2values:
            conc2values[x].append(y)
        else:
            conc2values[x] = [y]

    for conc, values in conc2values.items():
        conc2values[conc] = numpy.mean(values)


    points = [(k, v) for k, v in conc2values.items()]
    points.sort(key=lambda x: x[0])

    base = (points[0][1]+points[1][1])/2
    toxic = []

    points = [(p[0], p[1]/base) for p in points]
    max_fold = max([p[1] for p in points])
    cutoff = 1+(max_fold-1)*0.5

    if points[-1][1] < cutoff:
        points = list(reversed(points))
        
        for j, p in enumerate(points):
            if points[j][1] <= cutoff:
                toxic.append(p[0])
            else:
                break
    
    if len(toxic) and len(toxic) < len(points):
        top = max_fold*base
        toxic_concs = [math.log10(v) for v in toxic]
        toxic_range = [min(toxic_concs) - 0.01, max(toxic_concs)+0.01]
        data_points = [v for v in data_points if v[0] not in toxic]
        data_points.append({'top': {'symbol': '<', 'right': top, 'left': ''}, "toxic_range": toxic_range})
    return data_points


def get_fnc(fnc, params):
    par_dict = {k: v for k, v in params.items()}
    return lambda x: fnc(x, **par_dict)



def compute_curves(data, params=None):
    if params is None:
        params = {}


    res = []
    fnc = curve_fncs[params["curve_equation"]]
    original_params = copy.deepcopy(params)

    for points in data:
        if params.get("response_change"):
            response_change = float(params["response_change"])
            rc_start = params["response_change_start"]
            rc_end = params["response_change_end"]
            rc_start = int(rc_start) if rc_start else False
            rc_end = int(rc_end) if rc_end else False

        
        if params.get("detect_toxicity"):
            points = detect_toxic_concentrations(points)
        

        params = copy.deepcopy(original_params)
        

        xy = [(math.log10(x), y) for x, y in points]
        xy.sort(key=lambda d: d[0])
        x = [val[0] for val in xy]
        y = [val[1] for val in xy]


        if numpy.isnan(x).any():
            res.append({
                "category": 5,
                "rejected": "error",
            })

        elif params.get("min_max_difference") and not evaluate_min_max_difference(xy, params["min_max_difference"]):
            res.append({
                "category": 5,
                "rejected": "minmax",
            })
        
        elif params.get("curve_slope") != "both" and evaluate_curve_slope(y) != params["curve_slope"]:
            res.append({
                "category": 5,
                "rejected": "slope",
            })

        elif params.get("response_change") and not evaluate_response_change(xy, response_change, rc_start, rc_end):
            res.append({
                "category": 5,
                "rejected": "response",
            })
        else:
            if params.get("smart_xc50"):
                params = get_smart_xc50(params, x, y)
            if params.get("smoothing"):
                x, y = smooth_curve(x ,y, 1)

            fitting_params = parse_fitting_params(params, x, y)

            start_conc = min(x)
            stop_conc = max(x)
            

            if len(x) > 4:
                fitter = Minimizer(error, fitting_params, fcn_args=(x, y, fnc), nan_policy="omit")
                # model = minimize(error, fitting_params, args=(x, y, fnc), method='least_squares')
                model = fitter.minimize(method="least_squares")
                dict_params = {p: model.params[p].value for p in model.params}

                res_fnc = get_fnc(fnc, dict_params)
                y_pred = [res_fnc(conc) for conc in x]
                # rmse = metrics.mean_squared_error(y, y_pred)
                
                if numpy.isnan(y_pred).any():
                    r2 = -1000
                    hcvalue = -1000
                    
                else:
                    r2 = round(metrics.r2_score(y, y_pred), 3)
                    hcvalue = res_fnc(stop_conc)

                step_count = 100
                step = abs(stop_conc - start_conc) / step_count
                xs = []
                ys = []
                for r in range(101):
                    x_val = start_conc + r*step
                    y_val = res_fnc(x_val)
                    xs.append(x_val)
                    ys.append(y_val)
                auc = metrics.auc(xs, ys)
                if math.isnan(auc):
                    auc = 0

                # fpr, tpr, thresholds = metrics.roc_curve(numpy.array(ys), numpy.array(xs), pos_label=2)


                r = {
                    "symmetry" : dict_params.get("symmetry"),
                    "slope": dict_params.get("slope"), 
                    "inflection": dict_params.get("inflection"), 
                    "bottom": dict_params["bottom"], 
                    "top": dict_params["top"], 
                    "r2": r2,
                    "concmin" : start_conc,
                    "concmax" : stop_conc,
                    "hcvalue" : hcvalue,
                    "equation" : params["curve_equation"],
                    "auc" : auc,
                }
                r["category"] = evaluate_curve(r, x, y)
                res.append(r)
    return res

def evaluate_curve_slope(y):
    part_size = int(len(y)*0.2)

    if numpy.mean(y[:part_size]) < numpy.mean(y[-part_size:]):
        return "ascending"
    else:
        return "descending"


def evaluate_min_max_difference(xy, diff):
    ys = [y for x, y in xy]
    ys.sort()

    if ys and ys[-1] - ys[0] <= float(diff):
        return False
    else:
        return True



def evaluate_response_change(xy, response_change, rc_start, rc_end):
    x2ys = {}

    for pair in xy:
        if pair[0] in x2ys:
            x2ys[pair[0]].append(pair[1])
        else:
            x2ys[pair[0]] = [pair[1]]
    xs = list(x2ys.keys())
    xs.sort()
    part_size = int(len(x2ys)*0.2)

    rc_start = rc_start if rc_start else part_size
    
    values = []
    for x in xs[:rc_start]:
        values.extend(x2ys[x])
    start = numpy.mean(values)

    rc_end = rc_end if rc_end else part_size
    values = []
    for x in xs[-rc_end:]:
        values.extend(x2ys[x])
    end = numpy.mean(values)
    
    if abs(end-start) < response_change:
        return False
    else:
        return True

def evaluate_curve(params, x, y):
    inflection = 1 if params["inflection"] < params["concmax"] and params["inflection"] > params["concmin"] else 0
    value_range = 1 if max(y) >= params["top"]*1.1 and min(y) <= params["bottom"]*1.1 else 0
    min_y = float(min(y))
    if min_y == 0:
        min_y = 0.01
        
    top_bottom_ratio = 1 if max(y)/min_y >= 2 else 0
    r2 = 1 if params["r2"] >= 0.5 else 0
    score = inflection + value_range + top_bottom_ratio + r2

    if score == 4:
        category = 1
    elif score == 3:
        category = 2
    elif score >= 1:
        category = 3
    else:
        category = 4
    
    return category

def get_curve_points(start_conc, stop_conc, fnc, step_count=100):
    res = []
    step = abs(stop_conc - start_conc) / step_count
    for i in range(step_count + 1):
        conc = start_conc + i * step
        res.append((conc, fnc(conc)))
    return res

def detect_toxicity(item):
    item.sort(key=lambda x: x[0])
    log.info(item)
    return

def get_smart_xc50(params, xs, ys):
    if params["inflection"].get("left") == None and params["inflection"].get("right") == None:
        params["inflection"] = {"symbol": "<x<", "left":min(xs)-1, "right":max(xs)+1}

    return params

def get_smart_top_bottom(params, xs, ys, controls):
    # control2type = {"high" :2,"low" : 3}
    margin_in = float(params["boundary_margin_in"]) if params["boundary_margin_in"] else 0
    margin_out = float(params["boundary_margin_out"]) if params["boundary_margin_out"] else 0

    highs = [float(x["2"]["average"]) if "2" in x else None for x in controls]
    lows = [float(x["3"]["average"]) if "3" in x else None for x in controls]

    # if ap_controls and ap_controls.get("high", False):
    if None not in highs:
        high = numpy.mean(highs)
        if margin_in or margin_out:
            params["top"] = {"symbol": "<x<", "right":high+margin_out, "left": high-margin_in}
        else:
            params["top"] = {"symbol": "<", "right":high}

    # if ap_controls and ap_controls.get("low", False):
    if None not in lows:
        low = numpy.mean(lows)
        if margin_in or margin_out:
            params["bottom"] = {"symbol": "<x<", "right":low+margin_in, "left": low-margin_out}
        else:
            params["bottom"] = {"symbol": ">", "right":low}
    return params

def smooth_curve(xs, ys, n=1):
    x2ys = {}
    for i, x in enumerate(xs):
        if x in x2ys:
            x2ys[x].append(ys[i])
        else:
            x2ys[x] = [ys[i]]
    
    xs_order = x2ys.keys()
    xs_order.sort()
    y_medians = []
    for x in xs_order:
        y_medians.append(numpy.mean(x2ys[x]))
    
    # y_with_boundaries = [y_medians[0] for i in range(n)]
    # y_with_boundaries.extend(y_medians)
    # y_with_boundaries.extend([y_medians[-1] for i in range(n)])

    medfilter = signal.medfilt(y_medians, 2*n+1)
    y_medfilter = [y_medians[0]]
    y_medfilter.extend(medfilter[1:-1])
    y_medfilter.append(y_medians[-1])

    return xs_order, y_medfilter

def parse_fitting_params(params, x, y):
    fitting_params = Parameters()
    param2value = {
        "top": numpy.max(y),
        "bottom": numpy.min(y),
        "slope": 1,
        "inflection": numpy.mean(x),
        "symmetry": 1,
    }

    if len(params) == 0:
        params = {
            'slope': {'symbol': '<', 'right': '', 'left': ''},
            'symmetry': {'symbol': '<', 'right': '', 'left': ''},
            'bottom': {'symbol': '<', 'right': '', 'left': ''},
            'top': {'symbol': '<', 'right': '', 'left': ''},
            'inflection': {'symbol': '<', 'right': '', 'left': ''},
        }

    parsed_params = {}
    eq = params.get("curve_equation", "4pl")
    if eq != "5pl" and "symmetry" in params:
        del params["symmetry"]
    if eq == "3pl" and "slope" in params:
        del params["slope"]
    for p, value in params.items():
        if p in param2value:
            current_params = {"value": param2value[p]}
            if p in params:
                right = value.get("right")
                left = value.get("left")
                if right is not None:
                    if value["symbol"] == "<":
                        current_params["max"] = right
                    elif value["symbol"] == ">":
                        current_params["min"] = right
                    elif value["symbol"] == "=":
                        current_params["value"] = right
                        current_params["vary"] = False
                    elif left is not None:
                        lr = [left, right]
                        lr.sort()
                        current_params["min"] = lr[0]
                        current_params["max"] = lr[1]

            # if value["right"] != "":
            #     if value["symbol"] == "<":
            #         current_params["max"] = float(value["right"])
            #     elif value["symbol"] == ">":
            #         current_params["min"] = float(value["right"])
            #     elif value["symbol"] == "=":
            #         current_params["value"] = float(value["right"])
            #         current_params["vary"] = False
            #     else:
            #         current_params["max"] = float(value["right"])
            #         if value["left"] != "":
            #             current_params["min"] = float(value["left"])
            # print(current_params)
            parsed_params[p] = current_params
    for p, val in parsed_params.items():
        fitting_params.add(p, **val)

    return fitting_params

def error(params, x, y, fnc):
    par_dict = {p: params[p].value for p in params}
    if params["top"] < params["bottom"]: # a try to restrain the algorithm from swapping top / bottom
        params["top"], params["bottom"] = params["bottom"], params["top"]
        if "slope" in params:
            params["slope"] * -1
    return numpy.array([fnc(x, **params) - y for x, y in zip(x, y)])

if __name__ == "__main__":
    res = compute_curves([
            [
                (7.5e-08, 0.968359684754773), (2.1256e-07, 1.00181670245553), (1.6251e-06, 0.224356550084932), (6.25e-07, 0.71612711127561), (2.52e-08, 1.05534470512703), (4.6251e-06, 0.0516341741044428), (7.5e-08, 1.06998377870389), (1.6251e-06, 0.198086114508497), (2.52e-08, 1.10541216462806), (2.1256e-07, 1.00404240075182), (6.25e-07, 0.734308196355223), (7.5e-08, 1.02603430147625), (4.6251e-06, 0.0525952522985339), (2.1256e-07, 1.04387627979203), (6.25e-07, 0.677526422930234), (1.6251e-06, 0.210596672825038), (2.52e-08, 0.984574777693805), (4.6251e-06, 0.0647108752986569),
            ],
            [
                (6.25e-07, 1.11287499470146), (7.5e-08, 1.136405695753), (1.625e-06, 1.03361375119282), (2.125e-07, 1.03447475153709), (2.5e-08, 1.02583083741795), (4.625e-06, 1.06071044934748), (6.25e-07, 1.17867535386515), (7.5e-08, 0.999330057369007), (2.5e-08, 1.01747185568939), (1.625e-06, 1.02934762268583), (2.125e-07, 1.04009689355459), (4.625e-06, 1.06973275698968), (7.5e-08, 1.05905916482553), (1.625e-06, 1.07260772067528), (6.25e-07, 0.905832119406935), (2.125e-07, 0.985407656989373), (4.625e-06, 1.00143045158062), (2.5e-08, 1.09077639940987),
            ],
            [
                (2.125e-07, 0.201586357983018), (6.25e-07, 0.0747711804740671), (2.125e-07, 0.207616668751622), (1.625e-06, 0.0505035425284338), (1.625e-06, 0.0552650977464908), (7.5e-08, 0.750066322252744), (2.5e-08, 0.969822806377275), (4.625e-06, 0.0489519223114675), (6.25e-07, 0.0992034092636111), (6.25e-07, 0.0890359959762088), (1.625e-06, 0.0539814545819217), (2.5e-08, 0.933906437549044), (7.5e-08, 0.609512771815265), (7.5e-08, 0.671016813699086), (4.625e-06, 0.0392948233491549), (4.625e-06, 0.0418542523264482), (2.5e-08, 0.989803225029388), (2.125e-07, 0.202281526848881),
            ]
        ],
        {   #mandaotry
            'curve_equation': '4pl',
            # curves params restrictions
            'bottom': {'right': 0.0, 'symbol': '>'},
            'top': {'left': 0.8, 'right': 1.2, 'symbol': '<x<'},
            'inflection': {'symbol': '<'},
            'slope': {'right': 15.0, 'symbol': '>'},
            'symmetry': {'symbol': '<'},
            #other stuff
            'curve_slope': 'descending',
            'detect_toxicity': False,
            'min_max_difference': '',
            'response_change': '0.3',
            'response_change_end': '',
            'response_change_start': '',
            'smart_xc50': True,
            'smoothing': False,
         }
    )
    from pprint import pprint
    pprint(res)