import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def compare_promo_distribution(train, test):
    """
    Function to compare the distribution of the 'Promo' column in training and test datasets
    and visualize them side-by-side.

    Parameters:
    - train_file: str, path to the training dataset CSV file.
    - test_file: str, path to the test dataset CSV file.
    """

    # Compare 'Promo' distribution in training and test sets
    train_promo_dist = train['Promo'].value_counts(normalize=True)
    test_promo_dist = test['Promo'].value_counts(normalize=True)

    # Display Promo distributions
    print("Training set Promo distribution:\n", train_promo_dist)
    print("Test set Promo distribution:\n", test_promo_dist)

    # Plotting the Promo distributions
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot Promo distributions for training and test sets
    train['Promo'].value_counts(normalize=True).plot(kind='bar', ax=ax[0], title="Training Set Promo", color=['blue', 'green'])
    test['Promo'].value_counts(normalize=True).plot(kind='bar', ax=ax[1], title="Test Set Promo", color=['blue', 'green'])

    # Set titles and labels
    ax[0].set_xlabel('Promo')
    ax[0].set_ylabel('Percentage')
    ax[1].set_xlabel('Promo')
    ax[1].set_ylabel('Percentage')

    plt.tight_layout()
    plt.show()


def analyze_sales_around_holidays(train_df):
    # Ensure 'Date' column is in datetime format
    train_df['Date'] = pd.to_datetime(train_df['Date'])

    # Sort by store and date
    train_df = train_df.sort_values(by=['Store', 'Date'])

    # Create flags for holiday-related periods
    train_df['BeforeHoliday'] = train_df['StateHoliday'].shift(-1).notna() & (train_df['StateHoliday'].shift(-1) != '0')
    train_df['AfterHoliday'] = train_df['StateHoliday'].shift(1).notna() & (train_df['StateHoliday'].shift(1) != '0')
    train_df['DuringHoliday'] = train_df['StateHoliday'].notna() & (train_df['StateHoliday'] != '0')

    # Assign each sale to before, during, or after holiday
    train_df['Period'] = np.where(train_df['BeforeHoliday'], 'Before Holiday',
                    np.where(train_df['DuringHoliday'], 'During Holiday',
                    np.where(train_df['AfterHoliday'], 'After Holiday', 'Regular Day')))

    # Group by period to get average sales for each category
    period_sales = train_df.groupby('Period')['Sales'].mean().reset_index()

    # Print the result
    print(period_sales)

    # Plotting the sales behavior
    plt.figure(figsize=(10,6))
    plt.bar(period_sales['Period'], period_sales['Sales'], color=['blue', 'green', 'red', 'gray'])
    plt.title("Average Sales Before, During, and After Holidays")
    plt.xlabel("Period")
    plt.ylabel("Average Sales")
    plt.show()

# Create a column for holiday classification
def classify_holiday(date):
    if date.month == 12 and date.day >= 20:
        return 'Christmas'
    elif date.month == 4 and date.day == 1:  # Adjust based on Easter date
        return 'Easter'
    else:
        return 'Regular Day'
    
 # Could the promos be deployed in more effective ways? Which stores should promos be deployed in?
def merge_data(train, store):
    """Merge training data with store data on 'Store' column."""
    merged_data = train.merge(store, on='Store', how='left')
    return merged_data

def analyze_promo_effectiveness(merged_data):
    """Analyze average sales and customers based on promo status."""
    promo_analysis = merged_data.groupby('Promo').agg(
        Average_Sales=('Sales', 'mean'),
        Average_Customers=('Customers', 'mean')
    ).reset_index()
    return promo_analysis

def identify_underperforming_stores(merged_data):
    """Identify stores with low total sales."""
    store_performance = merged_data.groupby('Store').agg(
        Total_Sales=('Sales', 'sum'),
        Average_Customers=('Customers', 'mean'),
        Promo_Usage=('Promo', 'mean')  # Average promo usage (0-1)
    ).reset_index()
    
    underperforming_stores = store_performance[store_performance['Total_Sales'] < store_performance['Total_Sales'].quantile(0.25)]
    return underperforming_stores

def calculate_sales_lift(merged_data):
    """Calculate sales lift during promotions for each store."""
    sales_lift = merged_data.groupby('Store').agg(
        Sales_Before_Promo=('Sales', lambda x: x[merged_data['Promo'] == 0].mean()),
        Sales_During_Promo=('Sales', lambda x: x[merged_data['Promo'] == 1].mean())
    ).reset_index()
    
    sales_lift['Sales_Lift'] = sales_lift['Sales_During_Promo'] - sales_lift['Sales_Before_Promo']
    return sales_lift

def analyze_store_type_performance(merged_data):
    """Analyze average sales and customers based on store type and assortment."""
    store_type_performance = merged_data.groupby(['StoreType', 'Assortment']).agg(
        Average_Sales=('Sales', 'mean'),
        Average_Customers=('Customers', 'mean')
    ).reset_index()
    return store_type_performance

def analyze_competition_distance(merged_data):
    """Analyze the impact of competition distance on sales and customers."""
    competition_analysis = merged_data.groupby('CompetitionDistance').agg(
        Average_Sales=('Sales', 'mean'),
        Average_Customers=('Customers', 'mean')
    ).reset_index()
    return competition_analysis

def analyze_customer_behavior_by_time(train):
    """
    Analyze customer behavior during store opening and closing times.
    
    Parameters:
    - train (DataFrame): The training data containing store information.
    
    Returns:
    - behavior_analysis (DataFrame): A DataFrame summarizing customer behavior.
    """
    
    # Ensure 'Date' column is in datetime format
    train['Date'] = pd.to_datetime(train['Date'])
    
    # Create a 'Hour' column from the 'Date' column
    train['Hour'] = train['Date'].dt.hour
    
    # Group by 'Hour' and aggregate customer counts
    behavior_analysis = train.groupby('Hour').agg(
        Average_Customers=('Customers', 'mean'),
        Total_Customers=('Customers', 'sum')
    ).reset_index()
    
    # Plotting the results
    plt.figure(figsize=(12, 6))
    plt.plot(behavior_analysis['Hour'], behavior_analysis['Average_Customers'], marker='o', color='b')
    plt.fill_between(behavior_analysis['Hour'], behavior_analysis['Average_Customers'], color='lightblue', alpha=0.5)
    plt.title('Average Customer Count by Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Customers')
    plt.xticks(range(0, 24))  # Set x-ticks for each hour
    plt.grid()
    plt.show()
    
    return behavior_analysis


def get_stores_open_all_weekdays(df):
    """
    Identifies stores that are open on all weekdays (Monday to Friday).
    
    Args:
        df (pd.DataFrame): The DataFrame containing store data.
    
    Returns:
        list: List of store IDs open all weekdays.
    """
    weekdays = [0, 1, 2, 3, 4]  # Monday to Friday
    open_stores = df[(df['Open'] == 1) & (df['DayOfWeek'].isin(weekdays))]
    
    # Check if the store was open on all weekdays (at least once for every weekday)
    store_open_counts = open_stores.groupby('Store')['DayOfWeek'].nunique()
    
    # Stores open all weekdays
    stores_open_weekdays = store_open_counts[store_open_counts == len(weekdays)].index.tolist()
    
    return stores_open_weekdays

def calculate_average_sales(df, stores, days):
    """
    Calculates the average sales for specific stores on specific days.
    
    Args:
        df (pd.DataFrame): The DataFrame containing store data.
        stores (list): List of store IDs to consider.
        days (list): List of days of the week to filter (e.g., [5, 6] for weekends).
        
    Returns:
        pd.Series: Average sales grouped by store.
    """
    filtered_sales = df[(df['Store'].isin(stores)) & (df['DayOfWeek'].isin(days)) & (df['Open'] == 1)]
    average_sales = filtered_sales.groupby('Store')['Sales'].mean()
    
    return average_sales

def compare_weekend_sales(df, stores_open_weekdays):
    """
    Compares the weekend sales of stores open all weekdays versus other stores.
    
    Args:
        df (pd.DataFrame): The DataFrame containing store data.
        stores_open_weekdays (list): List of store IDs open all weekdays.
        
    Returns:
        pd.DataFrame: A DataFrame with average weekend sales for both store categories.
    """
    # Weekend days: Saturday (5) and Sunday (6)
    weekend_days = [5, 6]
    
    # Calculate weekend sales for stores open on all weekdays
    avg_sales_open_weekdays = calculate_average_sales(df, stores_open_weekdays, weekend_days)
    
    # Calculate weekend sales for other stores (those not open all weekdays)
    other_stores = df[~df['Store'].isin(stores_open_weekdays)]['Store'].unique()
    avg_sales_other_stores = calculate_average_sales(df, other_stores, weekend_days)
    
    # Combine the results into a DataFrame for easy comparison
    sales_comparison = pd.DataFrame({
        'Stores Open Weekdays': avg_sales_open_weekdays,
        'Other Stores': avg_sales_other_stores
    })
    
    return sales_comparison


def calculate_sales_by_assortment(df):
    """
    Calculates average sales by assortment type.
    
    Args:
        df (pd.DataFrame): The merged dataset containing sales and assortment information.
    
    Returns:
        pd.DataFrame: A DataFrame with average sales per assortment type.
    """
    # Calculate the average sales for each assortment type
    sales_by_assortment = df[df['Open'] == 1].groupby('Assortment')['Sales'].mean().reset_index()
    sales_by_assortment.columns = ['Assortment', 'Average Sales']
    
    return sales_by_assortment

def analyze_sales_by_competition_distance(train_df, store_df):
    """
    Analyzes the impact of competition distance on average sales.

    Parameters:
    - train_df: DataFrame containing training data with sales information.
    - store_df: DataFrame containing store information including competition distance.

    Returns:
    - DataFrame containing average sales grouped by competition distance range.
    """

    # Step 1: Calculate the average sales for each store
    avg_sales_per_store = train_df[train_df['Open'] == 1].groupby('Store')['Sales'].mean().reset_index()

    # Step 2: Retrieve competition distance from 'store_df'
    store_competition_distance = store_df[['Store', 'CompetitionDistance']]

    # Step 3: Categorize competition distance into bins and label them
    bins = [0, 500, 1000, 5000, np.inf]
    labels = ['<500m', '500-1000m', '1000-5000m', '>5000m']
    store_competition_distance['CompetitionDistanceRange'] = pd.cut(store_competition_distance['CompetitionDistance'], bins=bins, labels=labels)

    # Step 4: Combine the two datasets using 'Store' as the key
    sales_with_competition_distance = pd.merge(avg_sales_per_store, store_competition_distance, on='Store', how='left')

    # Step 5: Calculate average sales per competition distance range
    avg_sales_by_comp_distance = sales_with_competition_distance.groupby('CompetitionDistanceRange')['Sales'].mean().reset_index()

    return avg_sales_by_comp_distance

def analyze_competitor_opening_effect(train_df, store_df):
    # Step 1: Get store IDs with NA in CompetitionDistance
    stores_with_na = store_df[store_df['CompetitionDistance'].isna()]['Store']
    print(f'Stores with NA in CompetitionDistance: {stores_with_na.tolist()}')

    # Step 2: Get entries for those stores where CompetitionDistance later has values
    affected_stores = store_df[store_df['Store'].isin(stores_with_na)]
    
    # Identify which of these stores later have valid CompetitionDistance
    stores_with_valid_distance = affected_stores[affected_stores['CompetitionDistance'].notna()]['Store'].unique()
    print(f'Stores that later have valid CompetitionDistance: {stores_with_valid_distance.tolist()}')

    # Step 3: Analyze sales impact
    results = {}
    for store in stores_with_valid_distance:
        sales_data = train_df[train_df['Store'] == store]
        average_sales = sales_data['Sales'].mean()
        results[store] = average_sales

    return results