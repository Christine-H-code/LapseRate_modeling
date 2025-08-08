import sys
sys.path.append("C:/Users/E1005279/OneDrive - Sanlam Life Insurance Limited/MWL/MWLrepo/MiWayLife2/Data_Analysis_Package")
from data_preparation import DataPreparation, Analysis, multi_data_ops
import data_operations as do

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
#________________________________________________________________________________________________________   

def res_code_categories(code): # creating buckets for resolution code to hot-one encode it and not have too many columns
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

def last_n_payments(group):
    sortby='collection_date'
    n=4
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
    calls.df['res_code']= calls.df['resolution_code'].apply(res_code_categories)

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

    #______________Remove calls related to cancellation resolution codes_______________________
    calls = DataPreparation(df=calls_data)
    calls.cleaning_ops(drop_nulls = 'N',filtering = 'Y',filter_conditions_exclude={'resolution_code':['retentions_policy_cancelled','policy_cancelled','unpaid_cancelled_policy']})
    calls.df['res_code']= calls.df['resolution_code'].apply(res_code_categories)
 #!!!!!!!!!!!!!!!!!!update res_code buckets!!!!!!!!!!!!!!!!!!

    #_______________Adding sales date to calls data to filter to calls made after sale______________________
    calls_sale_date = multi_data_ops(file_list=[],df1=calls.df, df2=policy_data)
    calls_sale_date.merged_df = calls_sale_date.merging(on='policy_name', df_left=calls_sale_date.df1, df_right=calls_sale_date.df2, how='left', col_right=['policy_name','first_month_at_risk'])

    #__________________Selecting only call records before inception date___________________
    calls.df = calls_sale_date.merged_df[calls_sale_date.merged_df['datetime_start'].dt.strftime('%Y-%m-%d')<calls_sale_date.merged_df['first_month_at_risk'].dt.strftime('%Y-%m-%d')]

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

    #_______________Splitiing claims and monthly premium payments____________
    claims_data = payment_data[payment_data['collection_sub_type']=='Claim']
    payment_data = payment_data[payment_data['collection_sub_type']=='Monthly']

    #_______________aggregating claims data to policy________________
    claims = DataPreparation(df=claims_data)
    agg_dict = {'# claims': pd.NamedAgg(column='collections_id', aggfunc=pd.Series.nunique)}
    claims.aggregation(groupby_cols=['policy_id'],cat_cols=[],agg_dict_not_cat_cols=agg_dict)
    claims.agg_df

    #________________making paid/unpaid flag__________________________
    payment_data['paid?']= payment_data['collected_amount'].apply(lambda x: 'paid' if pd.notnull(x) else 'unpaid')

    #________________adding previous n months payment status______________
    last_month_payments_data = payment_data.groupby('policy_id', as_index=False).apply(last_n_payments)

    #_________________Adding duration to next Aniversary column__________________
    payment_data['duration to anniversary (months)'] = payment_data.apply(
        lambda row: ((row['anniversary_due'] - row['cancellation_effective_date']) / np.timedelta64(1, 'D'))//30
        if pd.notnull(row['cancellation_effective_date'])
        else ((row['anniversary_due'] - pd.Timestamp('2025-08-01')) / np.timedelta64(1, 'D'))//30,
        axis=1)

    #________________aggregating payments data_____________________
    payments = DataPreparation(df=payment_data)
    agg_dict={'sales_channel': pd.NamedAgg(column='sales_channel', aggfunc='first'),
              '* policy start delay months': pd.NamedAgg(column='* policy start delay months', aggfunc='max'),
              '* policy duration months': pd.NamedAgg(column='* policy duration months', aggfunc='max'),
              'last premium amount due': pd.NamedAgg(column='amount', aggfunc='last'),
              'duration to anniversary': pd.NamedAgg(column='duration to anniversary', aggfunc='max')}
    payments.aggregation(groupby_cols=['policy_id'],cat_cols=['paid?','payment_method'],agg_dict_not_cat_cols=agg_dict)
    payments.agg_df

    #__________________number of anniversaries in policy lifetime_____________
    payments.agg_df['# anniversaries']=payments.agg_df['* policy duration months']//12

    #___________________portion of payment due that were paid_______________
    payments.agg_df['payment rate']=payments.agg_df['paid?_paid_sum']/(payments.agg_df['paid?_paid_sum']+payments.agg_df['paid?_unpaid_sum'])

    #____________________Adding all data together (agg paments data, last months paid data and claims)________________________
    payments_data = multi_data_ops(file_list=[],df1=payments.agg_df,df2=last_month_payments_data,df3=claims.agg_df)
    pay_last = payments_data.merging(on='policy_id',df_left=payments_data.df1, df_right=payments_data.df2, how='left',col_left=['policy_id', 'sales_channel', '* policy start delay months',
            '* policy duration months', 'last premium amount due',
            'duration to anniversary','payment_method_DebiCheck_sum', 'payment_method_EFT_sum',
            'payment_method_Pre Fund_sum', '# anniversaries', 'payment rate'])
    payments_data.merged_df = payments_data.merging(on='policy_id',df_left=pay_last, df_right=payments_data.df3, how='left')
    payments_data.merged_df['# claims'] = payments_data.merged_df['# claims'].apply(lambda x: x if pd.notnull(x) else 0)

    return payments_data.merged_df
#____________________________________________________________________________________________________________________________________   

def lapses(lapse_data):
    lapse = DataPreparation(df=lapse_data)
    agg_dict={'cover_amount_full': pd.NamedAgg(column='cover_amount_full', aggfunc='mean'),
              'current_policy_status': pd.NamedAgg(column='current_policy_status', aggfunc='last'),
              'education': pd.NamedAgg(column='education', aggfunc='last'),
              'eml': pd.NamedAgg(column='eml', aggfunc='mean'),
              'pml': pd.NamedAgg(column='pml', aggfunc='mean'),
              'expected_lapse_rate': pd.NamedAgg(column='expected_lapse_rate', aggfunc='mean'),
              'gender': pd.NamedAgg(column= 'gender', aggfunc='last'),
              'hiv_group': pd.NamedAgg(column='hiv_group', aggfunc='last'),
              'current individual_income': pd.NamedAgg(column='individual_income', aggfunc='last'),
              'orginal individual_income': pd.NamedAgg(column='individual_income', aggfunc='first'),
              'lapse_type': pd.NamedAgg(column='lapse_type', aggfunc=lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan),
              'occupation_class': pd.NamedAgg(column='occupation_class', aggfunc='last'),
              'partner_income': pd.NamedAgg(column='partner_income', aggfunc='last'),
              'smoker_status': pd.NamedAgg(column='smoker_status', aggfunc='last')
              }
    lapse.aggregation(groupby_cols=['policy_name'],cat_cols=[],agg_dict_not_cat_cols=agg_dict)
    return lapse.agg_df

