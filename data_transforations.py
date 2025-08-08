import sys
sys.path.append("C:/Users/E1005279/OneDrive - Sanlam Life Insurance Limited/MWL/MWLrepo/MiWayLife2/Data_Analysis_Package")
from data_preparation import DataPreparation, Analysis, multi_data_ops
import data_operations as do

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
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

    policy_data['first_month_at_risk'] = pd.to_datetime(policy_data['first_month_at_risk'], errors='coerce')

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
              'duration to anniversary': pd.NamedAgg(column='duration to anniversary (months)', aggfunc='max')}
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

#________________________________________________________________________________________________________

def at_inception_data_merge(pre_inception_calls_data, sales_data,policy_data,lapse_data):

    at_inception_data = multi_data_ops(file_list=[],df1=pre_inception_calls_data, df2=sales_data,df3=policy_data,df4=lapse_data)

    sale_policy = at_inception_data.merging(on='policy_name',df_left=at_inception_data.df2, df_right=at_inception_data.df3, how='inner', col_left=['policy_id', 'policy_name', 'age', 'benefits_count', 
        'education','gender', 'income', 'lead_provider_name', 'lead_type', 'occupation',
        'payment_profile', 'sales_channel', 'smoker_status',
        'sold_socio_economic_class', 'underwriting_outcome'], col_right=['securitygroup_id', 'policy_name', 'campaign_name', 'optionality',
        'hiv_test_required', 'policy_type', 'policy_status', 'premium', 'cover_start_date_original','cover_start_date', 
        'cancellation_effective_date','sale_date', 'original_premium',
        'cancellation_reason', 'anniversary_due', 'last_benefit_amount',
        'first_month_at_risk', 'last_month_at_risk', 'last_settled', 'SEC',
        'pricing_version', 'first_collected_date',
        'number_of_collection_attempts', 'number_of_successful_collections',
        'total_collected',  'net_collected',
        'last_main_premium', 'last_benefit_type', 'total_funeral_premium',
        'funeral_count', 'has_been_recaptured', 'reason', 'last_benefit_status', 'prev_premium', 'fixed_debit_day',
        'lapse_flag'])
    
    sale_policy_lapse = at_inception_data.merging(on='policy_name',df_left=sale_policy, df_right=at_inception_data.df4, how='inner', col_right=['policy_name', 'cover_amount_full', 'current_policy_status',
         'eml', 'pml', 'expected_lapse_rate', 'hiv_group',
        'current individual_income', 'orginal individual_income', 'lapse_type',
        'occupation_class', 'partner_income'])
    
    at_inception_data.df_merged = at_inception_data.merging(on='policy_name',df_left=sale_policy_lapse, df_right=at_inception_data.df1, how='left')

    #making missing values 0 for specified columns
    for col in ['# calls',
          '# calls contacted', 'res_code_QA_sum', 'res_code_claims_sum',
          'res_code_no interest_sum', 'res_code_other_sum',
          'res_code_payment_sum', 'res_code_quote_sum', 'res_code_sale_sum',
          'res_code_update_sum','total_funeral_premium', 'funeral_count']:
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
    at_inception_data.df_merged = at_inception_data.df_merged[['policy_id', 'policy_name', 'age', 'education',
           'gender', 'income', 'smoker_status','sold_socio_economic_class','occupation_class','occupation','orginal individual_income','partner_income',
           'lead_provider_name', 'lead_type', 'sales_channel','sale_date', 'campaign_name',
           'underwriting_outcome','optionality', 'hiv_test_required','pricing_version', 'eml', 'pml','hiv_group',
           'securitygroup_id','policy_type','benefits_count','last_benefit_amount','cover_amount_full','total_funeral_premium','funeral_count','original_premium','fixed_debit_day','duration_policy_start_delay',
           '# calls','# calls contacted', 'res_code_QA_sum', 'res_code_claims_sum','res_code_no interest_sum', 'res_code_other_sum',
           'res_code_payment_sum', 'res_code_quote_sum', 'res_code_sale_sum','res_code_update_sum',
           'policy_status','current_policy_status','cancellation_effective_date', 'cancellation_reason', 'lapse_type', 'lapse_flag',  'expected_lapse_rate'
           , 'cover_start_date' ]] # use cover_start_date, cancellation_effective_date to get econimic indexes
    
    return at_inception_data.df_merged

#________________________________________________________________________________________________________

def post_sale_data_merge(post_sale_calls_data, sales_data,policy_data,lapse_data,payment_hist_data):

    post_sale_data = multi_data_ops(file_list=[],df1=post_sale_calls_data, df2=sales_data,df3=policy_data,df4=payment_hist_data, df5=lapse_data)

    sales_policy = post_sale_data.merging(on='policy_name',df_left=at_inception_data.df2, df_right=at_inception_data.df3, how='inner', col_left=['policy_id', 'policy_name', 'age', 'benefits_count', 
        'education','gender', 'income', 'lead_provider_name', 'lead_type', 'occupation',
        'payment_profile', 'sales_channel', 'smoker_status',
        'sold_socio_economic_class', 'underwriting_outcome'], col_right=['securitygroup_id', 'policy_name','campaign_name','optionality',
        'hiv_test_required', 'policy_type', 'policy_status', 'premium', 'cover_start_date_original','cover_start_date', 
        'cancellation_effective_date','sale_date', 'original_premium',
        'cancellation_reason', 'anniversary_due', 'last_benefit_amount',
        'first_month_at_risk', 'last_month_at_risk', 'last_settled', 'SEC',
        'pricing_version', 'first_collected_date',
        'number_of_collection_attempts', 'number_of_successful_collections',
        'total_collected',  'net_collected',
        'last_main_premium', 'last_benefit_type', 'total_funeral_premium',
        'funeral_count', 'has_been_recaptured', 'reason', 'last_benefit_status', 'prev_premium', 'fixed_debit_day',
        'lapse_flag'])
    
    sale_policy_lapse = post_sale_data.merging(on='policy_name',df_left=sales_policy, df_right=post_sale_data.df5, how='inner', col_right=['policy_name', 'cover_amount_full', 'current_policy_status',
         'eml', 'pml', 'expected_lapse_rate', 'hiv_group',
        'current individual_income', 'orginal individual_income', 'lapse_type',
        'occupation_class', 'partner_income'])
    
    sale_policy_lapse_pay = post_sale_data.merging(on='policy_id',df_left=sale_policy_lapse, df_right=post_sale_data.df4, how='inner', col_right=['policy_id', '* policy start delay months',
        '* policy duration months', 'last premium amount due',
        'duration to anniversary', 'payment_method_DebiCheck_sum',
        'payment_method_EFT_sum', 'payment_method_Pre Fund_sum',
        '# anniversaries', 'payment rate', '1 month ago payment',
        '2 month ago payment', '3 month ago payment', '4 month ago payment',
        '# claims'])
    
    post_sale_data.df_merged = post_sale_data.merging(on='policy_name',df_left=sale_policy_lapse_pay, df_right=post_sale_data.df1, how='left')

    #making missing values 0 for specified columns
    for col in ['# calls', '# calls contacted',
       'res_code_QA_sum', 'res_code_claims_sum', 'res_code_other_sum',
       'res_code_payment_sum', 'res_code_sale_sum', 'res_code_update_sum','total_funeral_premium', 'funeral_count']:
        post_sale_data.df_merged[col] = post_sale_data.df_merged[col].apply(lambda x: x if pd.notnull(x) else 0)

        post_sale_data.df_merged = post_sale_data.df_merged[['policy_id', 'policy_name', 'age','education','gender', 'income','current individual_income', 'orginal individual_income', 'partner_income','occupation','occupation_class', 'smoker_status',
             'sold_socio_economic_class',
             'lead_provider_name', 'lead_type','sales_channel', 'campaign_name',
             '# calls', '# calls contacted','res_code_QA_sum', 'res_code_claims_sum', 'res_code_other_sum','res_code_payment_sum', 'res_code_sale_sum', 'res_code_update_sum',
              'securitygroup_id','benefits_count', 'policy_type', 'original_premium','premium','last_main_premium', 'prev_premium','last premium amount due',
              '* policy start delay months', '* policy duration months','duration to anniversary','cover_amount_full','last_benefit_amount','fixed_debit_day', 'total_funeral_premium', 'funeral_count',
            '# anniversaries','has_been_recaptured', 'reason', 
            'underwriting_outcome','optionality', 'hiv_test_required','pricing_version', 'eml', 'pml','hiv_group',
            'number_of_collection_attempts', 'number_of_successful_collections','total_collected', 'net_collected', 'payment_method_DebiCheck_sum', 'payment_method_EFT_sum','payment_method_Pre Fund_sum',
            'payment rate','1 month ago payment', '2 month ago payment', '3 month ago payment', '4 month ago payment', '# claims',
            'policy_status','last_benefit_status', 'current_policy_status','lapse_type','cancellation_effective_date',  'cancellation_reason',  'lapse_flag', 'expected_lapse_rate'
               ,'cover_start_date']]
        
        return post_sale_data.df_merged
      



