
CREATE INDEX idx_sk_id_curr ON ln_application_data(SK_ID_CURR);


--Checking duplicate values, none found
SELECT count(SK_ID_CURR) FROM ln_application_data GROUP BY SK_ID_CURR having count(SK_ID_CURR)>1




-- There are missing values in NAME_TYPE_SUITE, which can be filled as "Unspecified"
select NAME_TYPE_SUITE, count(SK_ID_CURR) from ln_application_data
group by NAME_TYPE_SUITE

update ln_application_data 
set NAME_TYPE_SUITE = 'Unspecified'
where NAME_TYPE_SUITE is NULL

-- For value count display, created a sp to avoid code redundancy
-- Code Gender

EXEC spValueCount 'ln_application_data','CODE_GENDER';

update ln_application_data 
set CODE_GENDER = 'Unspecified'
where CODE_GENDER = 'XNA'

-- OWN_CAR_AGE is null where a customer don't own a car, so imputing the age as -1 to imply that the customer don't own a car
select FLAG_OWN_CAR, count(SK_ID_CURR) 
from ln_application_data
where OWN_CAR_AGE is NULL
group by FLAG_OWN_CAR

ALTER TABLE ln_application_data
ALTER COLUMN OWN_CAR_AGE smallint;

update ln_application_data
set OWN_CAR_AGE = -1
where OWN_CAR_AGE IS NULL AND FLAG_OWN_CAR = 0

select *
from ln_application_data
where OWN_CAR_AGE is NULL and FLAG_OWN_CAR = 1

-- Maybe the car could be new, so imputing 0
update ln_application_data
set OWN_CAR_AGE = 0
where OWN_CAR_AGE IS NULL AND FLAG_OWN_CAR = 1

-- OCCUPATION_TYPE
EXEC spValueCount 'ln_application_data','OCCUPATION_TYPE';

-- Lets see how many missing occupuation type applications were rejected
select TARGET, count(SK_ID_CURR)
from ln_application_data
where OCCUPATION_TYPE is Null
group by TARGET

-- MAjority of customer who has not specified their Occupation Type didn't face difficulties in repaying, so it much be important parameter for approving a loan
-- Hence, imputing missing occupation type as unknown

--Updated the occuputaion with relevant imputation
Update ln_application_data
Set OCCUPATION_TYPE = 'Retired'
where OCCUPATION_TYPE = 'Unknown' and NAME_INCOME_TYPE = 'Pensioner';

Update ln_application_data
Set OCCUPATION_TYPE = 'Retired'
where OCCUPATION_TYPE = 'Unknown' and NAME_INCOME_TYPE = 'Pensioner';

Update ln_application_data
Set OCCUPATION_TYPE = 'Government Roles'
where OCCUPATION_TYPE = 'Unknown' and NAME_INCOME_TYPE = 'State servant';

Update ln_application_data
Set OCCUPATION_TYPE = 'Unemployed'
where OCCUPATION_TYPE = 'Unknown' and NAME_INCOME_TYPE in ('Student', 'Unemployed');

Update ln_application_data
Set OCCUPATION_TYPE = 'On Leave'
where OCCUPATION_TYPE = 'Unknown' and NAME_INCOME_TYPE = 'Maternity leave';


-- CNT_FAM_MEMBERS
EXEC spValueCount 'ln_application_data','CNT_FAM_MEMBERS';

Update ln_application_data
Set CNT_FAM_MEMBERS = 0 
where CNT_FAM_MEMBERS is Null

-- Dropping all Client's Region details as there are alot of missing values
ALTER TABLE ln_application_data
DROP COLUMN 
APARTMENTS_AVG,
BASEMENTAREA_AVG,
YEARS_BEGINEXPLUATATION_AVG,
YEARS_BUILD_AVG,
COMMONAREA_AVG,
ELEVATORS_AVG,
ENTRANCES_AVG,
FLOORSMAX_AVG,
FLOORSMIN_AVG,
LANDAREA_AVG,
LIVINGAPARTMENTS_AVG,
LIVINGAREA_AVG,
NONLIVINGAPARTMENTS_AVG,
NONLIVINGAREA_AVG,
APARTMENTS_MODE,
BASEMENTAREA_MODE,
YEARS_BEGINEXPLUATATION_MODE,
YEARS_BUILD_MODE,
COMMONAREA_MODE,
ELEVATORS_MODE,
ENTRANCES_MODE,
FLOORSMAX_MODE,
FLOORSMIN_MODE,
LANDAREA_MODE,
LIVINGAPARTMENTS_MODE,
LIVINGAREA_MODE,
NONLIVINGAPARTMENTS_MODE,
NONLIVINGAREA_MODE,
APARTMENTS_MEDI,
BASEMENTAREA_MEDI,
YEARS_BEGINEXPLUATATION_MEDI,
YEARS_BUILD_MEDI,
COMMONAREA_MEDI,
ELEVATORS_MEDI,
ENTRANCES_MEDI,
FLOORSMAX_MEDI,
FLOORSMIN_MEDI,
LANDAREA_MEDI,
LIVINGAPARTMENTS_MEDI,
LIVINGAREA_MEDI,
NONLIVINGAPARTMENTS_MEDI,
NONLIVINGAREA_MEDI,
FONDKAPREMONT_MODE,
HOUSETYPE_MODE,
TOTALAREA_MODE,
WALLSMATERIAL_MODE,
EMERGENCYSTATE_MODE

--Circle observation, for the same record all the 4 columns are empty, so imputing 0
Update ln_application_data
Set OBS_30_CNT_SOCIAL_CIRCLE = 0,
OBS_60_CNT_SOCIAL_CIRCLE = 0,
DEF_30_CNT_SOCIAL_CIRCLE = 0,
DEF_60_CNT_SOCIAL_CIRCLE = 0
where OBS_30_CNT_SOCIAL_CIRCLE is NULL

-- DAYS_LAST_PHONE_CHANGE
select * from ln_application_data
where DAYS_LAST_PHONE_CHANGE is Null
 --Only one missing record and the customer has not stuggled in repaying, so imputing 0
 Update ln_application_data
Set DAYS_LAST_PHONE_CHANGE = 0
where DAYS_LAST_PHONE_CHANGE is NULL;

SELECT 'ALTER TABLE ln_application_data ALTER COLUMN ' 
       + COLUMN_NAME + ' tinyint NOT NULL;' AS AlterStatement
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'ln_application_data' and DATA_TYPE = 'bit';

ALTER TABLE ln_application_data ALTER COLUMN TARGET tinyint  NULL;
ALTER TABLE ln_application_data ALTER COLUMN FLAG_OWN_CAR tinyint  NULL;
ALTER TABLE ln_application_data ALTER COLUMN FLAG_OWN_REALTY tinyint  NULL;
ALTER TABLE ln_application_data ALTER COLUMN FLAG_EMP_PHONE tinyint  NULL;
ALTER TABLE ln_application_data ALTER COLUMN FLAG_WORK_PHONE tinyint  NULL;
ALTER TABLE ln_application_data ALTER COLUMN FLAG_CONT_MOBILE tinyint  NULL;
ALTER TABLE ln_application_data ALTER COLUMN FLAG_PHONE tinyint  NULL;
ALTER TABLE ln_application_data ALTER COLUMN FLAG_EMAIL tinyint  NULL;
ALTER TABLE ln_application_data ALTER COLUMN REG_REGION_NOT_LIVE_REGION tinyint  NULL;
ALTER TABLE ln_application_data ALTER COLUMN REG_REGION_NOT_WORK_REGION tinyint  NULL;
ALTER TABLE ln_application_data ALTER COLUMN LIVE_REGION_NOT_WORK_REGION tinyint  NULL;
ALTER TABLE ln_application_data ALTER COLUMN REG_CITY_NOT_LIVE_CITY tinyint  NULL;
ALTER TABLE ln_application_data ALTER COLUMN REG_CITY_NOT_WORK_CITY tinyint  NULL;
ALTER TABLE ln_application_data ALTER COLUMN LIVE_CITY_NOT_WORK_CITY tinyint  NULL;
ALTER TABLE ln_application_data ALTER COLUMN FLAG_DOCUMENT_3 tinyint  NULL;
ALTER TABLE ln_application_data ALTER COLUMN FLAG_DOCUMENT_5 tinyint  NULL;
ALTER TABLE ln_application_data ALTER COLUMN FLAG_DOCUMENT_6 tinyint  NULL;
ALTER TABLE ln_application_data ALTER COLUMN FLAG_DOCUMENT_8 tinyint  NULL;
ALTER TABLE ln_application_data ALTER COLUMN FLAG_DOCUMENT_13 tinyint  NULL;
ALTER TABLE ln_application_data ALTER COLUMN FLAG_DOCUMENT_14 tinyint  NULL;
ALTER TABLE ln_application_data ALTER COLUMN FLAG_DOCUMENT_16 tinyint  NULL;
ALTER TABLE ln_application_data ALTER COLUMN FLAG_DOCUMENT_18 tinyint  NULL;


-- Lets deal with missing external_sources rating
-- The goal is risk aversion we will average all the external source and get one rating

select sum(case when EXT_SOURCE_1 is null then 0 else 1 end) '1',
sum(case when EXT_SOURCE_2 is null then 0 else 1 end) '2',
sum(case when EXT_SOURCE_3 is null then 0 else 1 end) '3'
from dbo.ln_application_data

with avg_score as (
select SK_ID_CURR, Round(Avg(Score), 4) 'Avg_External_Rating' from (
select 
SK_ID_CURR, EXT_SOURCE_1 'Score'
from dbo.ln_application_data
where EXT_SOURCE_1 IS NOT NULL
union all
select SK_ID_CURR,
EXT_SOURCE_2
from dbo.ln_application_data
where EXT_SOURCE_2 IS NOT NULL
union all
select  SK_ID_CURR,
EXT_SOURCE_3
from dbo.ln_application_data
where EXT_SOURCE_3 IS NOT NULL
) a
group by Sk_id_Curr
)


--Alter Table dbo.ln_application_data
--Add avg_external_rating float

Update main
Set main.avg_external_rating = avg_table.Avg_External_Rating
from dbo.ln_application_data as main
join avg_score as avg_table
on main.SK_ID_CURR = avg_table.SK_ID_CURR;

select top 10 * from ln_application_data;

Select distinct Organization_type from ln_application_data

select ORGANIZATION_TYPE, Count(*) from ln_application_data 
group  by ORGANIZATION_TYPE


select count(*) from ln_application_data
where ORGANIZATION_TYPE = 'XNA'



-- For all retired people changing Organization type to NA

Begin TRANSACTION;

UPDATE ln_application_data
SET ORGANIZATION_TYPE = 'NA'
WHERE ORGANIZATION_TYPE = 'XNA';

select count(*) from ln_application_data where ORGANIZATION_TYPE = 'NA';

COMMIT;


Select top 10 *  from ln_application_data
where DAYS_REGISTRATION  is NULL


EXEC sp_rename 'dbo.ln_application_data.avg_external_rating', 'AVERAGE_EXTERNAL_RATING', 'COLUMN';


select TOP 100 * from dbo.ln_application_data
where AMT_ANNUITY is null











