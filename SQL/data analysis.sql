--Overall Repayment Difficulty Rate:
select Target, count(SK_ID_CURR) 'tot_cnt', 100 * count(SK_ID_CURR)/(select count(*) from ln_application_data) 'Percentage'
from ln_application_data
group by Target

--Good Loan vs. At-Risk Loan Ratio:


--Average Income of At-Risk Clients:
select Target, Round(AVG(AMT_INCOME_TOTAL), 2) 'Average Income'
from ln_application_data
group by Target;

-- Repeat Application 
select * from previous_application
where SK_ID_CURR in (select SK_ID_CURR from ln_application_data where target =1)

--Repeat Difficulty Rate:
select count(distinct SK_ID_CURR) from ln_application_data
