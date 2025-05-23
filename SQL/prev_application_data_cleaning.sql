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


select * from ln_application_data
where SK_ID_CURR = 113919