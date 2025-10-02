def key_metrics_monthly2(df, df2, df3, start_date=None, end_date=None, monthly_totals=None):
    """
    Create a pivoted dataframe with monthly KPIs, optionally including projected consultant value.

    Parameters
    ----------
    df : pandas DataFrame
        Invoice dataframe.
    df2 : pandas DataFrame
        Payments dataframe.
    df3 : pandas DataFrame
        Time-reporting dataframe.
    start_date : str, optional
        Start date filter in 'YYYY-MM-DD'.
    end_date : str, optional
        End date filter in 'YYYY-MM-DD'.
    monthly_totals : pandas DataFrame, optional
        Consultant projections with columns ['month', 'total_consultant_value'].

    Returns
    -------
    pandas.DataFrame
        Pivoted monthly KPI table.
    """
    plot_df = df.copy()
    payments_df = df2.copy()

    plot_df['final_pay_date'] = pd.to_datetime(plot_df['final_pay_date'])
    payments_df['final_pay_date'] = pd.to_datetime(payments_df['final_pay_date'])

    start_date_dt = pd.to_datetime(start_date) if start_date else None
    end_date_dt = pd.to_datetime(end_date) if end_date else None

    plot_df = plot_df.dropna(subset=['final_pay_date'])
    if start_date_dt is not None:
        plot_df = plot_df[plot_df['final_pay_date'] >= start_date_dt]
    if end_date_dt is not None:
        plot_df = plot_df[plot_df['final_pay_date'] <= end_date_dt]

    payments_df = payments_df.dropna(subset=['final_pay_date'])
    if start_date_dt is not None:
        payments_df = payments_df[payments_df['final_pay_date'] >= start_date_dt]
    if end_date_dt is not None:
        payments_df = payments_df[payments_df['final_pay_date'] <= end_date_dt]

    if plot_df.empty:
        return pd.DataFrame()

    plot_df['month'] = plot_df['final_pay_date'].dt.to_period('M')
    plot_df['month_str'] = plot_df['month'].dt.strftime('%Y-%m')

    plot_df['broker_type'] = plot_df['broker'].apply(
        lambda x: 'Broker' if 'broker' in str(x).lower()
        else ('Partner' if 'partner' in str(x).lower() else 'Direct')
    )

    monthly_invoice_totals = plot_df.groupby('month_str').agg(
        total_net=('invoice_amount_net', 'sum'),
        total_total=('invoice_amount_total', 'sum')
    )

    broker_data = plot_df.groupby(['month_str', 'broker_type']).agg(
        net_amount=('invoice_amount_net', 'sum'),
        total_amount=('invoice_amount_total', 'sum')
    ).reset_index()

    broker_data = broker_data.merge(monthly_invoice_totals, on='month_str')
    broker_data['pct_of_net'] = (broker_data['net_amount'] / broker_data['total_net']) * 100

    pivot_rows = []
    net_series = plot_df.groupby('month_str')['invoice_amount_net'].sum()
    total_series = plot_df.groupby('month_str')['invoice_amount_total'].sum()
    pivot_rows.append(('Net Amount', net_series))
    pivot_rows.append(('Total Invoice Amount', total_series))

    for broker_type in ['Broker', 'Direct', 'Partner']:
        subset = broker_data[broker_data['broker_type'] == broker_type]
        if subset.empty:
            continue
        subset = subset.set_index('month_str')
        pivot_rows.append((f'{broker_type} Net Amount', subset['net_amount']))
        pivot_rows.append((f'{broker_type} Total Amount', subset['total_amount']))
        pivot_rows.append((f'{broker_type} % of Net', subset['pct_of_net']))

    if monthly_totals is not None and not monthly_totals.empty:
        projected_df = monthly_totals.copy()
        projected_df['month'] = pd.to_datetime(projected_df['month'])
        if start_date_dt is not None:
            projected_df = projected_df[projected_df['month'] >= start_date_dt]
        if end_date_dt is not None:
            projected_df = projected_df[projected_df['month'] <= end_date_dt]
        if not projected_df.empty:
            projected_df['month_str'] = projected_df['month'].dt.strftime('%Y-%m')
            projected_series = projected_df.groupby('month_str')['total_consultant_value'].sum()
            pivot_rows.append(('Projected Net Amount', projected_series))

    payments_df['month_str'] = payments_df['final_pay_date'].dt.strftime('%Y-%m')
    payments_by_month = payments_df.groupby('month_str')['invoice_payment'].sum()
    pivot_rows.append(('Payments', payments_by_month))

    revenue_series = net_series.subtract(payments_by_month, fill_value=0)
    pivot_rows.append(('Revenue', revenue_series))

    if df3 is not None:
        time_df = df3.copy()
        time_df = time_df.dropna(subset=['date'])
        time_df['date'] = pd.to_datetime(time_df['date'])
        if start_date_dt is not None:
            time_df = time_df[time_df['date'] >= start_date_dt]
        if end_date_dt is not None:
            time_df = time_df[time_df['date'] <= end_date_dt]
        time_df['month_str'] = time_df['date'].dt.strftime('%Y-%m')
        hours_by_month = time_df.groupby('month_str').agg(
            billable_hours=('billable_hours', 'sum'),
            non_billable_hours=('non_billable_hours', 'sum'),
            total_hours=('total_hours', 'sum')
        )
        hours_by_month['Utilization Percentage'] = (
            (hours_by_month['billable_hours'] / hours_by_month['total_hours'])
            .replace([np.inf, -np.inf], 0)
            .fillna(0) * 100
        )
        pivot_rows.append(('Billable Hours', hours_by_month['billable_hours']))
        pivot_rows.append(('Non-Billable Hours', hours_by_month['non_billable_hours']))
        pivot_rows.append(('Total Hours', hours_by_month['total_hours']))
        pivot_rows.append(('Utilization Percentage', hours_by_month['Utilization Percentage']))

    result = pd.DataFrame({row_name: series for row_name, series in pivot_rows}).fillna(0)

    percentage_rows = ['Broker % of Net', 'Direct % of Net', 'Partner % of Net', 'Utilization Percentage']
    for row in percentage_rows:
        if row in result.columns:
            result[row] = result[row].apply(lambda x: f"{int(round(x))}%")

    for column in result.columns:
        if column not in percentage_rows:
            result[column] = result[column].apply(lambda x: f"{int(round(x)):,}" if not isinstance(x, str) else x)

    result = result.T
    result = result.reindex(sorted(result.columns), axis=1)

    formatted_cols = {}
    for col in result.columns:
        try:
            date_obj = pd.to_datetime(col[:7] + '-01')
            formatted_cols[col] = date_obj.strftime('%b %Y')
        except Exception:
            formatted_cols[col] = col
    result = result.rename(columns=formatted_cols)

    return result

# first version only with weighted amount filter 
def sales_funnel_viz(df, weighted_amount=True):
    """
    Sales funnel visualization using Plotly
    
    Arguments:
    df (DataFrame): Sales pipeline dataframe
    weighted_amount (bool): If True, use weighted_deal_amount, otherwise use deal_amount
    
    Returns:
    None: Displays the plotly figure
    """
    # select the appropriate amount column based on the parameter
    amount_col = 'weighted_deal_amount' if weighted_amount else 'deal_amount'
    amount_title = 'Weighted Deal Amount' if weighted_amount else 'Deal Amount'
    
    # group data by pipeline stage
    funnel_data = df.groupby(['pipeline_stage_order', 'pipeline_stage']).agg(
        amount=(amount_col, 'sum'),
        deal_count=('deal_id', 'count')
    ).reset_index().sort_values('pipeline_stage_order')
    
    # create stage labels
    funnel_data['stage_label'] = funnel_data.apply(
        lambda x: f"Stage {int(x['pipeline_stage_order'])}: {x['pipeline_stage']}", axis=1
    )
    
    # format amount for hover text
    funnel_data['amount_fmt'] = funnel_data['amount'].apply(lambda x: f"{x:,.2f} SEK")
    
    # create hover text
    funnel_data['hover_text'] = funnel_data.apply(
        lambda x: f"<b>{x['stage_label']}</b><br>" +
                 f"{amount_title}: {x['amount_fmt']}<br>" +
                 f"Number of Deals: {x['deal_count']}", 
        axis=1
    )
    
    # create funnel chart
    fig = go.Figure()
    fig.add_trace(go.Funnel(
        name='Sales Funnel',
        y=funnel_data['stage_label'],
        x=funnel_data['amount'],
        textposition="inside",
        textinfo="value+percent initial",
        opacity=0.7,
        marker={
            "color": ["#1f77b4", "#7fc3e5", "#4292c6", "#2171b5", "#08519c"][:len(funnel_data)],
            "line": {"width": [1] * len(funnel_data), "color": ["white"] * len(funnel_data)}
        },
        hovertext=funnel_data['hover_text'],
        hoverinfo='text'
    ))
    
    # add title and labels
    fig.update_layout(
        title={
            'text': f'Sales Pipeline Funnel ({amount_title})',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
        },
        height=500,
        width=800
    )
    
    return fig
    
# second version with weighted amount and time range filter 
def sales_funnel_viz2(df, weighted_amount=True, time_filter_start=None, time_filter_stop=None):
    """
    Sales funnel visualization using Plotly with optional time filtering.
    
    Arguments:
    df (DataFrame): Sales pipeline dataframe
    weighted_amount (bool): If True, use weighted_deal_amount, otherwise use deal_amount
    time_filter_start (str): Start date for filtering in format 'YYYY-MM-DD' 
    time_filter_stop (str): End date for filtering in format 'YYYY-MM-DD' 
    
    Returns:
    None: Displays the plotly figure
    """
    # create a copy of the dataframe to avoid modifying the original
    filtered_df = df.copy()
    
    # apply time filters if provided
    if time_filter_start or time_filter_stop:
        # get a sample date to check timezone info
        filtered_df['create_date']
        
        # alternative approach for filtering with timezone-aware dates
        if time_filter_start:
            # convert to datetime and localize to match dataframe timezone
            start_date = pd.to_datetime(time_filter_start)
            start_date_str = start_date.strftime('%Y-%m-%d')
            # Create a mask by converting to string format first for comparison
            filtered_df = filtered_df[filtered_df['create_date'].dt.strftime('%Y-%m-%d') >= start_date_str]
        
        if time_filter_stop:
            # convert to datetime and localize to match dataframe timezone
            end_date = pd.to_datetime(time_filter_stop)
            end_date_str = end_date.strftime('%Y-%m-%d')
            # Create a mask by converting to string format first for comparison
            filtered_df = filtered_df[filtered_df['create_date'].dt.strftime('%Y-%m-%d') <= end_date_str]
        
        # create date range text for title
        if time_filter_start and time_filter_stop:
            date_range_text = f" (Created {time_filter_start} to {time_filter_stop})"
        elif time_filter_start:
            date_range_text = f" (Created from {time_filter_start})"
        elif time_filter_stop:
            date_range_text = f" (Created until {time_filter_stop})"
    else:
        date_range_text = ""
    
    # check if we have data after filtering
    if filtered_df.empty:
        print("No data available after applying time filters.")
        return
    
    # select the appropriate amount column based on the parameter
    amount_col = 'weighted_deal_amount' if weighted_amount else 'deal_amount'
    amount_title = 'Weighted Deal Amount' if weighted_amount else 'Deal Amount'
    
    # group data by pipeline stage
    funnel_data = filtered_df.groupby(['pipeline_stage_order', 'pipeline_stage']).agg(
        amount=(amount_col, 'sum'),
        deal_count=('deal_id', 'count')
    ).reset_index().sort_values('pipeline_stage_order')
    
    # create stage labels
    funnel_data['stage_label'] = funnel_data.apply(
        lambda x: f"Stage {int(x['pipeline_stage_order'])}: {x['pipeline_stage']}", axis=1
    )
    
    # format amount for hover text
    funnel_data['amount_fmt'] = funnel_data['amount'].apply(lambda x: f"{x:,.2f} SEK")
    
    # create hover text
    funnel_data['hover_text'] = funnel_data.apply(
        lambda x: f"<b>{x['stage_label']}</b><br>" +
                 f"{amount_title}: {x['amount_fmt']}<br>" +
                 f"Number of Deals: {x['deal_count']}", 
        axis=1
    )
    
    # create funnel chart
    fig = go.Figure()
    fig.add_trace(go.Funnel(
        name='Sales Funnel',
        y=funnel_data['stage_label'],
        x=funnel_data['amount'],
        textposition="inside",
        textinfo="value+percent initial",
        opacity=0.7,
        marker={
            "color": ["#1f77b4", "#7fc3e5", "#4292c6", "#2171b5", "#08519c"][:len(funnel_data)],
            "line": {"width": [1] * len(funnel_data), "color": ["white"] * len(funnel_data)}
        },
        hovertext=funnel_data['hover_text'],
        hoverinfo='text'
    ))
    
    # add title and labels
    fig.update_layout(
        title={
            'text': f'Sales Pipeline Funnel ({amount_title}){date_range_text}',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
        },
        height=500,
        width=800
    )    
    return fig


# apply color MoM color trend and percentage formatting to pivoted dataframe 
def highlight_revenue_trend(row, color=True):
    # color rows
    target_rows = [
        'Net Amount',
        'Payments',
        'Revenue',
        'Broker % of Net',
        'Direct % of Net',
        'Partner % of Net',
        'Utilization Percentage',
        'Billable Hours',
        'Non-Billable Hours',
        'Total Hours'
    ]
    # percentage rows
    percentage_rows = [
        'Broker % of Net',
        'Direct % of Net',
        'Partner % of Net',
        'Utilization Percentage'
    ]
    # convert all values to int and add %
    if row.name in percentage_rows:
        formatted = []
        for v in row:
            try:
                val = float(str(v).replace('%', '').replace(',', ''))
                formatted.append(f"{int(round(val))}%")
            except Exception:
                formatted.append(v)
        row[:] = formatted

    if row.name not in target_rows:
        return [''] * len(row)
    # convert values to int (remove commas and % if present)
    values = []
    for v in row:
        try:
            values.append(int(str(v).replace(',', '').replace('%', '')))
        except Exception:
            values.append(0)
    # compare each month to previous (first column has no previous month)
    styles = ['']
    for i in range(1, len(values)):
        if not color:
            styles.append('')
        elif values[i] > values[i-1]:
            styles.append('color: green; font-weight: bold;')
        elif values[i] < values[i-1]:
            styles.append('color: red; font-weight: bold;')
        else:
            styles.append('')
    return styles

