import sys
sys.path.append("C:/Users/E1005279/OneDrive - Sanlam Life Insurance Limited/MWL/MWLrepo/MiWayLife2/Data_Analysis_Package")
from data_preparation import DataPreparation, Analysis, multi_data_ops
import data_operations as do

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
#________________________________________________________________________________________________________   

def res_code_categories_postsale(code): # creating buckets for resolution code to hot-one encode it and not have too many columns
    if code in ['debit_date_changed', 'cover_reduced','requested_changes_completed','banking_details_changed','alternative_payer','cover_increased','commencement_moved',
                'additional_lives_update','bmi_update','banking_details_update', 'beneficiary_update','occupation_update','personal_details_update',
                 'educational_level_update','income_brackets_update','alcohol_intake_update']:
        return "update"
    elif code in ['claim_no_docs','claim_no_claim','claim_repudiated', 'claim','claim_accepted']:
        return "claims"
    elif code in['qa_no_risk','qa_risk_absorbed',  'qa_recaptured']:
        return "QA"
    elif code in ['sent_to_retentions', 'free_cover','premium_holiday_added','unpaid_policy_reduced', 'second_debit_scheduled','once_off_payment', 'policy_retained', 'unpaid_collect_again',
                  'policy_reinstated', 'policy_disputed','double_debit_scheduled']:
        return "payment"
    elif code in ['completed', 'no_contact_post_sale','product_explanation', 'sum_assured_disclosed', 'confirm_acceptance','converted' ]:
        return "sale"
    elif code in ['duplicate', 'no_changes_requested','unreachable_no_answer', 'testing', 'auto_close', 'sale','sysytem_duplicate', 'cannot_offer_cover_not_the_life_insured', 'no_sale_web_preferred',
       'not_interested_no_reason','not_interested_unreachable_after_initial_contact','not_interested_does_not_need_cover', 'unreachable_bulk_close',
       'unconverted',  'unreachable_voicemail_only','web_quote_declined', 'no_sale_quote_expired',  'not_interested_in_a_will','not_interested_competitors_already_covered', 
       'reading_script','note_on_behalf_of_sales_agent','unreachable_hung_up',  'cannot_offer_cover_age_too_old','no_sale_unreachable_after_initial_contact', 'documents_posted',
       'web_quote_not_interested_after_follow_up', 'not_interested_irate_do_not_call', 'no_sale_price_affordability']:
        return "other"
    else:
        return np.nan
    
#________________________________________________________________________________________________________   

def res_code_categories(code):
    if code in ['debit_date_changed','cover_reduced','requested_changes_completed','banking_details_changed',
       'commencement_moved','policy_disputed','cover_increased','additional_lives_update', 'bmi_update', 'banking_details_update','personal_details_update', 'income_brackets_update',
       'beneficiary_update','occupation_update', 'educational_level_update','alcohol_intake_update', 'documents_posted']:
        return "update"
    elif code in ['claim_accepted','claim','claim_repudiated', 'claim_no_claim','claim_no_docs']:
        return "claims"
    elif code in['qa_risk_absorbed','qa_recaptured','qa_no_risk']:
        return "QA"
    elif code in ['policy_cancelled', 'once_off_payment','alternative_payer','policy_retained','unpaid_collect_again','unpaid_policy_reduced',
        'second_debit_scheduled','policy_reinstated','double_debit_scheduled','free_cover', 'premium_holiday_added','sent_to_retentions']:
        return "payment"
    elif code in ['sale', 'no_sale_product_funeral_only','no_contact_post_sale','converted', 'product_explanation','sum_assured_disclosed','confirm_acceptance']:
        return "sale"
    elif code in ['web_quote_web_preferred','quoted_before']:
        return "quote"
    elif code in ['web_quote_declined','not_interested_does_not_need_cover', 'no_sale_uw_declined','not_interested_no_reason', 'no_sale_price_affordability','no_sale_unreachable_after_initial_contact',
       'web_quote_not_interested_after_follow_up','no_sale_product_Benefits_not_enough', 'not_interested_quick_quote_was_too_much','no_sale_price_not_competitive','web_quote_expired','not_interested_competitors_took_offer_from_first_contact',
       'not_interested_competitors_already_covered','not_interested_unreachable_after_initial_contact', 'not_interested_brand_negative_old_mutual','not_interested_affiliate_interested_in_competition',
       'not_interested_competitors_specific_provider','cannot_offer_cover_income_not_enough','not_interested_affiliate_cant_recall', 'no_sale_quote_expired','not_interested_affiliate_didnt_ask','not_interested_web_preferred','not_interested_hiv_test_do_not_want_to_go',
       'not_interested_brand_dont_know','no_sale_uw_exclusions','no_sale_web_preferred','not_interested_irate_do_not_call','not_interested_affiliate_did_not_opt_in','no_sale_product_retrenchment',
       'cannot_offer_cover_age_too_old', 'no_sale_product_short-term_insurance','no_sale_product_retirement_annuity','not_interested_waiting_for_loan_approval', 'unconverted',
       'cannot_offer_cover_citizenship_not_sa','not_interested_in_a_will', 'do_not_call', 'not_interested_cannot_afford','cannot_offer_cover_language_-_cannot_speak_english',
       'cannot_offer_cover_pensioner','quoted_not_interested','cannot_offer_cover_not_the_life_insured','unemployed']:
        return "no interest"
    elif code in ['completed',  'unreachable_hung_up','unreachable_voicemail_only', 'unreachable_no_answer','price_beat_case_created','sysytem_duplicate','testing', 'unreachable_wrong_number','duplicate', 'no_sale_timing_start_later','uw_manual','no_changes_requested', 
       'auto_close_external', 'uw_rule_change','auto_close', 'unreachable_bulk_close','invalidate_review','inbound_queries','existing_client','reading_script','note_on_behalf_of_sales_agent','busy']:
        return "other"
    else:
        return np.nan  

#________________________________________________________________________________________________________   

def last_n_payments(group):
    sortby='collection_date'
    n=6
    new_val = 'paid?'
    sorted_group = group.sort_values(sortby,ascending=False)
    top_n = sorted_group[new_val].head(n).tolist()
    top_n += ['N/A']*(n-len(top_n))
    new_cols=[]
    for m in range(0,n):
        new_cols.append(str(m+1)+' month ago payment')
    return pd.Series(top_n, index=new_cols)
#________________________________________________________________________________________________________   

def post_sale_calls(calls_data,policy_data):

    #______________Remove calls related to cancellation resolution codes_______________________
    calls = DataPreparation(df=calls_data)
    calls.cleaning_ops(drop_nulls = 'N',filtering = 'Y',filter_conditions_exclude={'resolution_code':['retentions_policy_cancelled','policy_cancelled','unpaid_cancelled_policy']})
    calls.df['res_code']= calls.df['resolution_code'].apply(res_code_categories_postsale)

    #_______________Adding sales date to calls data to filter to calls made after sale______________________
    calls_sale_date = multi_data_ops(file_list=[],df1=calls.df, df2=policy_data)
    calls_sale_date.merged_df = calls_sale_date.merging(on='policy_name', df_left=calls_sale_date.df1, df_right=calls_sale_date.df2, how='left', col_right=['policy_name','sale_date'])

    #__________________Selecting only call records after sale date___________________
    calls.df = calls_sale_date.merged_df[calls_sale_date.merged_df['datetime_start'].dt.strftime('%Y-%m-%d')>calls_sale_date.merged_df['sale_date'].dt.strftime('%Y-%m-%d')]

    #__________________Aggregating calls data to policy_name_____________________
    agg_dict={'# calls': pd.NamedAgg(column='call_id', aggfunc=pd.Series.nunique),
          '# calls contacted': pd.NamedAgg(column='contact_indicator', aggfunc='sum')
          }
    calls.aggregation(groupby_cols=['policy_name'],cat_cols=['res_code'] ,agg_dict_not_cat_cols=agg_dict)
    
    return calls.agg_df
#________________________________________________________________________________________________________   

def pre_sale_calls(calls_data,policy_data):

    policy_data['sale_date'] = pd.to_datetime(policy_data['sale_date'], errors='coerce')

    #______________Remove calls related to cancellation resolution codes_______________________
    calls = DataPreparation(df=calls_data)
    calls.cleaning_ops(drop_nulls = 'N',filtering = 'Y',filter_conditions_exclude={'resolution_code':['retentions_policy_cancelled','policy_cancelled','unpaid_cancelled_policy']})
    calls.df['res_code']= calls.df['resolution_code'].apply(res_code_categories)

    #_______________Adding sales date to calls data to filter to calls made after sale______________________
    calls_sale_date = multi_data_ops(file_list=[],df1=calls.df, df2=policy_data)
    calls_sale_date.merged_df = calls_sale_date.merging(on='policy_name', df_left=calls_sale_date.df1, df_right=calls_sale_date.df2, how='left', col_right=['policy_name','sale_date'])

    #__________________Selecting only call records before inception date___________________
    
    calls.df = calls_sale_date.merged_df[calls_sale_date.merged_df['datetime_start'].dt.strftime('%Y-%m-%d')<calls_sale_date.merged_df['sale_date'].dt.strftime('%Y-%m-%d')]

    #__________________Aggregating calls data to policy_name_____________________
    agg_dict={'# calls': pd.NamedAgg(column='call_id', aggfunc=pd.Series.nunique),
          '# calls contacted': pd.NamedAgg(column='contact_indicator', aggfunc='sum')
          }
    calls.aggregation(groupby_cols=['policy_name'],cat_cols=['res_code'] ,agg_dict_not_cat_cols=agg_dict)
    
    return calls.agg_df
#________________________________________________________________________________________________________   

def payments_history(payment_data,current_date='2025-08-01'):

    #_______________Exclude payments for current month_______________
    payment_data = payment_data[payment_data['collection_date']< pd.Timestamp(current_date)]
    
    #_______________making anniversary date a datetime dtype_____________
    payment_data['anniversary_due'] = pd.to_datetime(payment_data['anniversary_due'], errors='coerce')

    #_______________Getting monthly premium payments____________

    payment_data = payment_data[payment_data['collection_sub_type']=='Monthly']


    #________________making paid/unpaid flag__________________________
    payment_data['paid?']= payment_data['collected_amount'].apply(lambda x: 'paid' if pd.notnull(x) else 'unpaid')

    #________________adding previous n months payment status______________
    last_month_payments_data = payment_data.groupby('policy_id', as_index=False).apply(last_n_payments)

    #_________________Adding duration to next Aniversary column__________________
    payment_data['duration to anniversary (months)'] = payment_data.apply(
        lambda row: ((row['anniversary_due'] - row['cancellation_effective_date']) / np.timedelta64(1, 'D'))//30
        if pd.notnull(row['cancellation_effective_date'])
        else ((row['anniversary_due'] - pd.Timestamp(current_date)) / np.timedelta64(1, 'D'))//30,
        axis=1)

    #________________aggregating payments data_____________________
    payments = DataPreparation(df=payment_data)
    agg_dict={'* policy start delay months': pd.NamedAgg(column='* policy start delay months', aggfunc='max'),
              '* policy duration months': pd.NamedAgg(column='* policy duration months', aggfunc='max'),
              'last premium amount due': pd.NamedAgg(column='amount', aggfunc='last'),
              'duration to anniversary': pd.NamedAgg(column='duration to anniversary (months)', aggfunc='max')}
    payments.aggregation(groupby_cols=['policy_id'],cat_cols=['paid?','payment_method'],agg_dict_not_cat_cols=agg_dict)
    payments.agg_df

    #__________________number of anniversaries in policy lifetime_____________
    payments.agg_df['# anniversaries']=payments.agg_df['* policy duration months']//12

    #___________________portion of payment due that were paid_______________
    payments.agg_df['payment rate']=payments.agg_df['paid?_paid_sum']/(payments.agg_df['paid?_paid_sum']+payments.agg_df['paid?_unpaid_sum'])

    #____________________Adding all data together (agg paments data, last months paid data)________________________
    payments_data = multi_data_ops(file_list=[],df1=payments.agg_df,df2=last_month_payments_data)
    payments_data.merged_df = payments_data.merging(on='policy_id',df_left=payments_data.df1, df_right=payments_data.df2, how='left',col_left=['policy_id', '* policy start delay months',
            '* policy duration months', 'last premium amount due',
            'duration to anniversary','payment_method_DebiCheck_sum', 'payment_method_EFT_sum',
            'payment_method_Pre Fund_sum', '# anniversaries', 'payment rate'])


    return payments_data.merged_df
#____________________________________________________________________________________________________________________________________   

def lapses(lapse_data):
    lapse = DataPreparation(df=lapse_data)
    agg_dict={'cover_amount_full': pd.NamedAgg(column='cover_amount_full', aggfunc='mean'),
              'eml': pd.NamedAgg(column='eml', aggfunc='mean'),
              'pml': pd.NamedAgg(column='pml', aggfunc='mean'),
              'current individual_income': pd.NamedAgg(column='individual_income', aggfunc='last'),
              'orginal individual_income': pd.NamedAgg(column='individual_income', aggfunc='first'),
              'lapse_type': pd.NamedAgg(column='lapse_type', aggfunc=lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan),
              'occupation_class': pd.NamedAgg(column='occupation_class', aggfunc='last')
              }
    lapse.aggregation(groupby_cols=['policy_name'],cat_cols=[],agg_dict_not_cat_cols=agg_dict)
    return lapse.agg_df

#________________________________________________________________________________________________________

def at_inception_data_merge(pre_inception_calls_data, sales_data,policy_data,lapse_data):


    at_inception_data = multi_data_ops(file_list=[],df1=pre_inception_calls_data, df2=sales_data,df3=policy_data,df4=lapse_data)

    sale_policy = at_inception_data.merging(on='policy_name',df_left=at_inception_data.df2, df_right=at_inception_data.df3, how='inner', col_left=['policy_id', 'policy_name', 'age', 'benefits_count', 
        'education','gender', 'income', 'lead_provider_name', 'lead_type',  'smoker_status',
        'sold_socio_economic_class'], col_right=[ 'policy_name', 'campaign_name', 'optionality',
        'hiv_test_required', 'policy_type', 'premium', 'cover_start_date_original','cover_start_date', 
        'cancellation_effective_date','sale_date', 'original_premium', 'last_benefit_amount',
        'total_funeral_premium','funeral_count', 'has_been_recaptured',   'prev_premium',
        'lapse_flag'])
    
    sale_policy_lapse = at_inception_data.merging(on='policy_name',df_left=sale_policy, df_right=at_inception_data.df4, how='inner', col_right=['policy_name', 'cover_amount_full',
         'eml', 'pml',  'current individual_income', 'orginal individual_income', 'lapse_type',
        'occupation_class'])
    
    at_inception_data.df_merged = at_inception_data.merging(on='policy_name',df_left=sale_policy_lapse, df_right=at_inception_data.df1, how='left')
 
    #making missing values 0 for specified columns
    for col in list(set(['# calls',
          '# calls contacted','total_funeral_premium', 'funeral_count'])|set([col for col in at_inception_data.df_merged.columns if col.startswith('res_code_')])):
        at_inception_data.df_merged[col] = at_inception_data.df_merged[col].apply(lambda x: x if pd.notnull(x) else 0)

    #Replace missing orginal_premium values with premium
    at_inception_data.df_merged['original_premium'] = at_inception_data.df_merged.apply(
        lambda row: row['original_premium'] 
        if pd.notnull(row['original_premium'])
        else row['premium'],
        axis=1)
    
    #addining new column: difference in the original and actual policy start date
    at_inception_data.df_merged['duration_policy_start_delay'] =  ((at_inception_data.df_merged[ 'cover_start_date'] - at_inception_data.df_merged['cover_start_date_original']) / np.timedelta64(1, 'D'))//30

    #ordering the columns
    cols=list(set(['policy_id', 'policy_name', 'age', 'education',
           'gender', 'income', 'smoker_status','sold_socio_economic_class','occupation_class','orginal individual_income',
           'lead_provider_name', 'lead_type', 'sale_date', 'campaign_name',
           'optionality', 'hiv_test_required', 'eml', 'pml','policy_type','benefits_count','last_benefit_amount','cover_amount_full','total_funeral_premium','funeral_count','original_premium','duration_policy_start_delay',
           '# calls','# calls contacted',
           'cancellation_effective_date', 'lapse_type', 'lapse_flag' , 'cover_start_date' ])|set([col for col in at_inception_data.df_merged.columns if col.startswith('res_code_')]))
    at_inception_data.df_merged = at_inception_data.df_merged[cols] # use cover_start_date, cancellation_effective_date to get econimic indexes
    
    return at_inception_data.df_merged

#________________________________________________________________________________________________________

def post_sale_data_merge(post_sale_calls_data, sales_data,policy_data,lapse_data,payment_hist_data):

    post_sale_data = multi_data_ops(file_list=[],df1=post_sale_calls_data, df2=sales_data,df3=policy_data,df4=payment_hist_data, df5=lapse_data)

    sales_policy = post_sale_data.merging(on='policy_name',df_left=post_sale_data.df2, df_right=post_sale_data.df3, how='inner', col_left=['policy_id', 'policy_name', 'age', 'benefits_count', 
        'education','gender', 'income', 'lead_provider_name', 'lead_type', 'smoker_status',
        'sold_socio_economic_class'], col_right=[ 'policy_name','campaign_name','optionality',
        'hiv_test_required', 'policy_type', 'premium', 'cover_start_date_original','cover_start_date', 
        'cancellation_effective_date','sale_date', 'original_premium','last_benefit_amount',
         'total_funeral_premium','funeral_count', 'has_been_recaptured', 'prev_premium',
        'lapse_flag'])
    
    sale_policy_lapse = post_sale_data.merging(on='policy_name',df_left=sales_policy, df_right=post_sale_data.df5, how='inner', col_right=['policy_name', 'cover_amount_full',
         'eml', 'pml',
        'current individual_income', 'orginal individual_income', 'lapse_type',
        'occupation_class'])
    
     
    
    sale_policy_lapse_pay = post_sale_data.merging(on='policy_id',df_left=sale_policy_lapse, df_right=post_sale_data.df4, how='inner')
    
    post_sale_data.df_merged = post_sale_data.merging(on='policy_name',df_left=sale_policy_lapse_pay, df_right=post_sale_data.df1, how='left')

    #making missing values 0 for specified columns
    for col in list(set(['# calls','# calls contacted','total_funeral_premium', 'funeral_count'])|set([col for col in post_sale_data.df_merged.columns if col.startswith('res_code_')])):
        post_sale_data.df_merged[col] = post_sale_data.df_merged[col].apply(lambda x: x if pd.notnull(x) else 0)

    
    return post_sale_data.df_merged
      

#________________________________________________________________________________________________________
def near_ftr_lapse_data_clean_filter(near_ftr_lapse_data):

    near_ftr_lapse_data.drop(columns=[ 'campaign_name','hiv_test_required',  'lead_provider_name', 'lead_type',
                                        'occupation_class','cover_start_date_original','sale_date'], inplace=True)


    near_ftr_lapse_data=near_ftr_lapse_data[near_ftr_lapse_data['has_been_recaptured']==0]
    near_ftr_lapse_data=near_ftr_lapse_data[near_ftr_lapse_data['policy_type']=='Fully Underwritten']
    return near_ftr_lapse_data


def at_inception_data_clean_filter(inception_data):


    inception_data=inception_data[inception_data['policy_type']=='Fully Underwritten']

    #exclude migrated policies
    inception_data = inception_data[~inception_data['campaign_name'].isin(['MWL_TE_IN Migrated Policies 2023','MWL_TE_OUT Migrated Policies 2023'])]
    return inception_data

#_______________________________________________________________________________________________________________________________

def adding_eco_ind(inflation, unemployment,data,date_col):
    data = pd.merge(data, inflation, left_on=date_col, right_on='year', how='left')
    data = pd.merge(data, unemployment, left_on=date_col, right_on='year', how='left')
    data.rename(columns={'ave':'unemployment rate'}, inplace=True)
    data.drop(columns=['year_x','year_y'], inplace=True)
    return data
#________________________________________________________________________________________________________________________________

def inception_targets(at_inception):
    at_inception['cover_start_date']= pd.to_datetime(at_inception['cover_start_date'])
    at_inception['cancellation_effective_date']= pd.to_datetime(at_inception['cancellation_effective_date'])
    current_date = datetime.strptime('2025-08-01', "%Y-%m-%d")
    at_inception['cancellation_effective_date']=pd.to_datetime(at_inception['cancellation_effective_date'])
    at_inception['end_date'] = at_inception['cancellation_effective_date'].apply(lambda x: x if pd.notnull(x) else current_date)
    at_inception['pol_duration'] = (at_inception['end_date']-at_inception['cover_start_date']).dt.days//30

    month3_df = at_inception.copy()
    month3_df['3month_lapse'] = ((at_inception['pol_duration']<4)&(at_inception['lapse_flag']==True)).astype(int)
    month3_df[['pol_duration','lapse_flag','3month_lapse']]


    month6_df = at_inception.copy()
    month6_df['6month_lapse'] = ((at_inception['pol_duration']<7)&(at_inception['lapse_flag']==True)).astype(int)
    month6_df[['pol_duration','lapse_flag','6month_lapse']]

    year1_df = at_inception.copy()
    year1_df['1yr_lapse'] = ((at_inception['pol_duration']<13)&(at_inception['lapse_flag']==True)).astype(int)
    year1_df[['pol_duration','lapse_flag','1yr_lapse']]

    year2_df = at_inception.copy()
    year2_df['2yr_lapse'] = ((at_inception['pol_duration']<25)&(at_inception['lapse_flag']==True)).astype(int)
    year2_df[['pol_duration','lapse_flag','2yr_lapse']]

    #flagging policy holder that lapse right after anniversay for year 1, 2,3 or 4
    ann_lapse_df = at_inception[at_inception['pol_duration']>11].copy()
    ann_lapse_df['ann_lapse'] = (((at_inception['pol_duration']>10)&(at_inception['pol_duration']<17)&(at_inception['lapse_flag']==True)) | 
                                 ((at_inception['pol_duration']>22)&(at_inception['pol_duration']<29)&(at_inception['lapse_flag']==True)) |
                                 ((at_inception['pol_duration']>34)&(at_inception['pol_duration']<41)&(at_inception['lapse_flag']==True))).astype(int)

    ann_lapse_df[['pol_duration','lapse_flag','ann_lapse']]

    at_inception_data = multi_data_ops(file_list=[],df1=at_inception,df2=month3_df,df3=month6_df,df4=year1_df,df5=ann_lapse_df)
    m_df=at_inception_data.merging(on='policy_id',df_left=at_inception_data.df1, df_right=at_inception_data.df2, how='left',col_right=['policy_id','3month_lapse'])
    m_df=at_inception_data.merging(on='policy_id',df_left=m_df, df_right=at_inception_data.df3, how='left',col_right=['policy_id','6month_lapse'])
    m_df=at_inception_data.merging(on='policy_id',df_left=m_df, df_right=at_inception_data.df4, how='left',col_right=['policy_id','1yr_lapse'])
    at_inception_data.df_merged=at_inception_data.merging(on='policy_id',df_left=m_df, df_right=at_inception_data.df5, how='left',col_right=['policy_id','ann_lapse'])

    at_inception_data = multi_data_ops(file_list=[],df1=at_inception_data.df_merged,df2=year2_df)
    at_inception_data.df_merged=at_inception_data.merging(on='policy_id',df_left=at_inception_data.df1, df_right=at_inception_data.df2, how='left',col_right=['policy_id','2yr_lapse'])
    return at_inception_data.df_merged

#________________________________________________________________________________________________________________

def near_ftr_targets(near_ftr):
    near_ftr['cover_start_date']= pd.to_datetime(near_ftr['cover_start_date'])
    near_ftr['cancellation_effective_date']= pd.to_datetime(near_ftr['cancellation_effective_date'])

    month3lapse= near_ftr[['lapse_flag','policy_id','lapse_type']].copy()
    month3lapse=month3lapse[month3lapse['lapse_type'].isin(['Payment Lapsed'])]
    month3lapse['payment lapse'] = month3lapse['lapse_flag']
    near_ftr = pd.merge(near_ftr,month3lapse[['policy_id','payment lapse']],how='left',on='policy_id')
    return near_ftr

#__________________________________________________________________________________________________________________

#gropuping function using decision trees



def group_by_decision_tree(model, column_data, column_name):
    """
    Groups a pandas Series into categorical bands based on the leaf nodes of a
    fitted Decision Tree Classifier model.

    This function first traverses the decision tree to determine the value ranges
    for each leaf node. It then applies the model to the input column data to get
    the leaf node index for each data point and uses the collected ranges to
    assign a categorical band.

    Args:
        model (sklearn.tree.DecisionTreeClassifier): A fitted decision tree model.
        column_data (pd.Series): The pandas Series to be grouped.
        column_name (str): The name of the column in the original DataFrame
                           that the decision tree was trained on.

    Returns:
        pd.Series: A new pandas Series containing the categorical bands.
    """
    # Ensure the model is a decision tree
    if not isinstance(model, DecisionTreeClassifier):
        raise TypeError("The provided model must be an instance of DecisionTreeClassifier.")
    
    tree = model.tree_
    
    # Check if the tree is empty or not fitted
    if tree.node_count == 0:
        raise ValueError("The decision tree model does not seem to be fitted.")

    # A helper dictionary to store the min/max range for each leaf node
    leaf_ranges = {}
    
    def traverse_tree(node_index, lower_bound=float('-inf'), upper_bound=float('inf')):
        """
        Recursively traverses the tree to find the value range for each leaf node.
        """
        # Base case: if it's a leaf node
        if tree.children_left[node_index] == tree.children_right[node_index]:
            leaf_ranges[node_index] = (lower_bound, upper_bound)
            return

        threshold = tree.threshold[node_index]
        left_child = tree.children_left[node_index]
        right_child = tree.children_right[node_index]

        # Recurse down the left child (less than or equal to threshold)
        traverse_tree(left_child, lower_bound, min(upper_bound, threshold))

        # Recurse down the right child (greater than threshold)
        traverse_tree(right_child, max(lower_bound, threshold), upper_bound)

    # Start the traversal from the root node (index 0)
    traverse_tree(0)

    # Use the model's apply method to get the leaf node for each data point
    # We need to reshape the data to a 2D array for the model
    leaf_indices = model.apply(column_data.values.reshape(-1, 1))

    # Create the new column of grouped bands
    grouped_bands = []
    for index in leaf_indices:
        lower, upper = leaf_ranges[index]

        # Format the string for the band
        if lower == float('-inf'):
            band = f"< {upper:.2f}"
        elif upper == float('inf'):
            band = f"> {lower:.2f}"
        else:
            band = f"{lower:.2f} - {upper:.2f}"
        
        grouped_bands.append(band)

    return pd.Series(grouped_bands, index=column_data.index, name=f'{column_name}_bands')