import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocessing_scaler(data):
    data['Date'] = pd.to_datetime(data['Date'])

    # Create a binary column for weekends
    data['IsWeekend'] = data['Date'].dt.dayofweek >= 5  # 5 for Saturday, 6 for Sunday

    # Create a binary column for state holidays
    data['IsHoliday'] = data['StateHoliday'].apply(lambda x: 1 if x in ['a', 'b', 'c'] else 0)

    # Sample list of holidays (You can modify it as needed)
    holidays = pd.to_datetime(['2024-01-01', '2024-04-07', '2024-12-25'])  # Add public holidays
    
    # Creating holiday and weekend ranges
    holiday_range = pd.date_range(start=data['Date'].min(), end=data['Date'].max(), freq='W-SAT').union(
        pd.date_range(start=data['Date'].min(), end=data['Date'].max(), freq='W-SUN')
    )

    # Convert holiday_range to a Series
    all_holidays = pd.Series(holiday_range)

    # Function to calculate days to the next holiday
    def calculate_days_to_holiday(date):
        next_holiday = all_holidays[all_holidays > date]
        return (next_holiday.min() - date).days if not next_holiday.empty else None

    # Function to calculate days after the last holiday
    def calculate_days_after_holiday(date):
        last_holiday = all_holidays[all_holidays < date]
        return (date - last_holiday.max()).days if not last_holiday.empty else None

    # Apply the functions
    data['DaysToHoliday'] = data['Date'].apply(calculate_days_to_holiday)
    data['DaysAfterHoliday'] = data['Date'].apply(calculate_days_after_holiday)

    return data