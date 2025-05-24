CREATE INDEX idx_prev_sk_id_curr ON previous_application(SK_ID_CURR);

select count(*) from previous_application 
where SK_ID_CURR in (select SK_ID_CURR from ln_application_data)

select sk_id_curr, count(*) from previous_application 
group by sk_id_curr
having count(*) > 1



select
--*
NAME_CONTRACT_STATUS, Name_Portfolio, count(*) 
from previous_application 
where DAYS_FIRST_DUE is null
group by NAME_CONTRACT_STATUS, Name_Portfolio

select
* 
from previous_application 
where DAYS_FIRST_DUE is null and NAME_CONTRACT_STATUS = 'Approved'

select
NAME_CLIENT_TYPE, NAME_CONTRACT_STATUS, COUNT(SK_ID_PREV) 
from previous_application 
GROUP BY NAME_CLIENT_TYPE, NAME_CONTRACT_STATUS
ORDER BY NAME_CLIENT_TYPE



SELECT DISTINCT NAME_YIELD_GROUP
FROM previous_application

select NAME_CONTRACT_TYPE, sum(AMT_DOWN_PAYMENT) 
from previous_application
GROUP BY  NAME_CONTRACT_TYPE


select * from
previous_application
where SK_ID_CURR = 117671

select  SK_ID_CURR, count(SK_ID_PREV)
--, NAME_CONTRACT_TYPE, AMT_ANNUITY, AMT_APPLICATION, AMT_CREDIT, AMT_GOODS_PRICE,
--AMT_DOWN_PAYMENT, WEEKDAY_APPR_PROCESS_START, HOUR_APPR_PROCESS_START, FLAG_LAST_APPL_PER_CONTRACT,
--NFLAG_LAST_APPL_IN_DAY, RATE_DOWN_PAYMENT, RATE_INTEREST_PRIMARY, RATE_INTEREST_PRIVILEGED,
--NAME_CASH_LOAN_PURPOSE, NAME_CONTRACT_STATUS, DAYS_DECISION, NAME_PAYMENT_TYPE, CODE_REJECT_REASON,
--NAME_TYPE_SUITE, NAME_CLIENT_TYPE, NAME_GOODS_CATEGORY, NAME_PORTFOLIO, NAME_PRODUCT_TYPE,
--CHANNEL_TYPE, NAME_SELLER_INDUSTRY, CNT_PAYMENT, NAME_YIELD_GROUP, PRODUCT_COMBINATION,
--DAYS_FIRST_DRAWING, DAYS_FIRST_DUE, DAYS_LAST_DUE_1ST_VERSION, DAYS_LAST_DUE, DAYS_TERMINATION,
--NFLAG_INSURED_ON_APPROVAL
 from previous_application
where SK_ID_CURR in (
select sk_id_curr from ln_application_data where Target = 1)
group by SK_ID_CURR
having count(SK_ID_PREV) > 1



select SK_ID_CURR, count(distinct sk_id_prev) 'TOT_PREV_APP',
SUM(Case when NAME_CONTRACT_STATUS = 'Approved' then 1 else 0 end) 'PREV_APPROVED_CNT',
SUM(Case when NAME_CONTRACT_STATUS = 'Canceled' then 1 else 0 end) 'PREV_CANCELLED_CNT',
SUM(Case when NAME_CONTRACT_STATUS = 'Refused' then 1 else 0 end) 'PREV_REFUSED_CNT',
SUM(Case when NAME_CONTRACT_STATUS = 'Unused offer' then 1 else 0 end) 'PREV_UNUSED_CNT'
from previous_application
where SK_ID_CURR in (
select sk_id_curr from ln_application_data)
Group by SK_ID_CURR








