import numpy as np
import pandas as pd

round_1_sp = pd.read_stata('data/NHATS_Round_1_SP_File.dta')

# Exclusions

# dementia
# (hc1disescn9) - 1 - YES, 2 - NO, -1 Inapplicable, -8 DK, -9 Missing
# filter hc1disescn9 == 2
# n post filter = 7146
round_1_cohort = round_1_sp[round_1_sp.hc1disescn9 == ' 2 NO']

# missing grip strength
# n post filter = 5969
# 2 measures, gr1grp1rdng and gr1grp2rdng
# remove those with na in both measurements
both_grip_na = [pd.isna(g1) and pd.isna(g2) for g1, g2 in zip(
    round_1_cohort.gr1grp1rdng, round_1_cohort.gr1grp2rdng)]
round_1_cohort = round_1_cohort[[not x for x in both_grip_na]]

# missing height or weight data
# weight in pounds (hw1currweigh), height in feet (hw1howtallft), height in inches (hw1howtallin)
# n post filter = 5822
missing_height_or_weight = [any([pd.isna(weight), pd.isna(height_f), pd.isna(height_in)]) for weight, height_f, height_in in zip(
    round_1_cohort.hw1currweigh, round_1_cohort.hw1howtallft, round_1_cohort.hw1howtallin)]
round_1_cohort = round_1_cohort[[not x for x in missing_height_or_weight]]

# Derived measures

# max grip_strength
# max of both grip readings (if applicable)
# appended as max_grip
round_1_cohort['max_grip'] = round_1_cohort.apply(
    lambda x: np.nanmax([x.gr1grp1rdng, x.gr1grp2rdng]), axis=1)

# BMI
# defined as self-reported baseline weight in kg divided by height in meters-squared
# appended as weight_kg, height_m, and BMI respectively
round_1_cohort['weight_kg'] = round_1_cohort.hw1currweigh.astype(
    'float') / 2.2046
round_1_cohort['height_m'] = (round_1_cohort.hw1howtallft.astype(
    'int') * 12 + round_1_cohort.hw1howtallin.astype('int')) * 0.0254
round_1_cohort['bmi'] = round_1_cohort.weight_kg / round_1_cohort.height_m**2

# High waist circumference
# waist measure in inches (wc1wstmsrinc)
# indicator for high waist circumference,  >= 102 cm in males, >= 88 cm in females
# appended as high_wc
# 104 missing wc measure


def high_wc(x):
    if pd.isna(x.r1dgender) or pd.isna(x.wc1wstmsrinc):
        return np.nan
    wc = x.wc1wstmsrinc * 2.54
    if x.r1dgender == '1 MALE':
        return True if wc >= 102 else False
    elif x.r1dgender == '2 FEMALE':
        return True if wc >= 88 else False
    else:
        raise Exception


round_1_cohort['high_wc'] = round_1_cohort.apply(high_wc, axis=1)


# Sarcopenia (defined by grip strength)
# grip strength < 35.5 kg in males, <20 kg in females
# appended as sarcopenia
# no na due to exclusion criteria

def sarcopenia(x):
    if pd.isna(x.max_grip) or pd.isna(x.r1dgender):
        return np.nan
    if x.r1dgender == '1 MALE':
        return True if x.max_grip < 35.5 else False
    elif x.r1dgender == '2 FEMALE':
        return True if x.max_grip < 20 else False
    else:
        raise Exception


round_1_cohort['sarcopenia'] = round_1_cohort.apply(sarcopenia, axis=1)


def sarcopenia_cutoff2(x):
    if pd.isna(x.max_grip) or pd.isna(x.r1dgender):
        return np.nan
    if x.r1dgender == '1 MALE':
        return True if x.max_grip < 26 else False
    elif x.r1dgender == '2 FEMALE':
        return True if x.max_grip < 16 else False
    else:
        raise Exception


round_1_cohort['sarcopenia_cutoff2'] = round_1_cohort.apply(
    sarcopenia_cutoff2, axis=1)

# SDOC Sarcopenia (defined by grip strength/BMI ratio)
# grip strength/BMI < 1.05 in males, < 0.79 in females
# appended as sdoc_sarcopenia
# no na due to exclusion criteria


def sdoc_sarcopenia(x):
    if any([pd.isna(m) for m in [x.max_grip, x.bmi, x.r1dgender]]):
        return np.nan

    ratio = x.max_grip / x.bmi
    if x.r1dgender == '1 MALE':
        return True if ratio < 1.05 else False
    elif x.r1dgender == '2 FEMALE':
        return True if ratio < 0.79 else False
    else:
        raise Exception


round_1_cohort['sdoc_sarcopenia'] = round_1_cohort.apply(
    sdoc_sarcopenia, axis=1)


# Gender
# r1dgender

round_1_cohort['gender'] = round_1_cohort.r1dgender

# Race
# rl1dracehisp, recode values below in dictionary
# no na
# appended as race


def race(x):
    d = {' 1 White, non-hispanic': 'White', ' 2 Black, non-hispanic': 'Black',
         ' 3 Other (Am Indian/Asian/Native Hawaiian/Pacific Islander/other specify), non-Hispanic': 'Other', ' 4 Hispanic': 'Hispanic', ' 5 more than one DKRF primary': 'Other', ' 6 DKRF': 'DKRF'}

    return d.get(x.rl1dracehisp, np.nan)


round_1_cohort['race'] = round_1_cohort.apply(race, axis=1)


# Smoking status
# Current - sd1smokedreg == 1 (smoked regularly) & sd1smokesnow == 1 (smokes now)
# Former smoker - sd1smokedreg == 1 & sd1smokesnow == 2 or sd1smokesnow is na
# Never - sd1smokedreg == 2 & sd1smokesnow == 2
# appended as smoking_status
# 1 overall na

def smoking_status(x):
    if pd.isna(x.sd1smokedreg) and pd.isna(x.sd1smokesnow):  # only 1
        return np.nan
    elif pd.isna(x.sd1smokedreg) and pd.notna(x.sd1smokesnow):  # never
        raise Exception
    elif pd.notna(x.sd1smokedreg) and pd.isna(x.sd1smokesnow):  # 2818
        if x.sd1smokedreg == ' 1 YES':
            return 'Former, maybe current'
        elif x.sd1smokedreg == ' 2 NO':
            return 'Never'
    else:  # both exist
        if x.sd1smokedreg == ' 1 YES' and x.sd1smokesnow == ' 1 YES':
            return 'Current'
        elif x.sd1smokedreg == ' 1 YES' and x.sd1smokesnow == ' 2 NO':
            return 'Former'
        else:
            return 'Never'


round_1_cohort['smoking_status'] = round_1_cohort.apply(smoking_status, axis=1)

# Education
# el1higstschl
# Less than high school: 1 - no schooling,
#                        2 - 1st to 8th grade,
#                        3 - 9th to 12th grade, no diploma
# High school to some college: 4 - high school graduate (diploma or equivalent)
#                              5 - vocational, technical, business or trade school certificate
#                                       beyond high school
#                              6 - some college but no degree
# College degree: 7 - associate's degree
#                 8 - bachelor's degree
# Graduate degree: 9 - master's, professional, or doctoral
# appended as education
# 4 na


def education(x):

    d = {' 1 NO SCHOOLING COMPLETED': 'Less than high school',
         ' 2 1ST-8TH GRADE': 'Less than high school',
         ' 3 9TH-12TH GRADE (NO DIPLOMA)': 'Less than high school',
         ' 4 HIGH SCHOOL GRADUATE (HIGH SCHOOL DIPLOMA OR EQUIVALENT)': 'High school to some college',
         ' 6 SOME COLLEGE BUT NO DEGREE': 'High school to some college',
         " 8 BACHELOR'S DEGREE": 'College degree',
         " 9 MASTER'S, PROFESSIONAL, OR DOCTORAL DEGREE": 'Graduate degree',
         ' 5 VOCATIONAL, TECHNICAL, BUSINESS, OR TRADE SCHOOL CERTIFICATE OR DIPLOMA (BEYOND HIGH SCHOOL LEVEL)': 'High school to some college',
         " 7 ASSOCIATE'S DEGREE": 'College degree'
         }

    return d.get(x.el1higstschl, np.nan)


round_1_cohort['education'] = round_1_cohort.apply(education, axis=1)

# Physical activity proxy
# pa1evrgowalk
# appended as ever_walk
# no na
round_1_cohort['ever_walk'] = round_1_cohort.apply(
    lambda x: True if x.pa1evrgowalk == ' 1 YES' else False, axis=1)

# Comorbidities

# heart_disease
# hc1disescn1 - had heart attack
# hc1disescn2 - has heart disease
# no na


def heart_disease(x):
    if x.hc1disescn1 == ' 1 YES' or x.hc1disescn2 == ' 1 YES':
        return True
    elif x.hc1disescn1 == ' 2 NO' or x.hc1disescn2 == ' 2 NO':
        return False
    else:
        return np.nan


round_1_cohort['heart_disease'] = round_1_cohort.apply(heart_disease, axis=1)

# hypertension
# hc1disescn3
# 7 na
round_1_cohort['hypertension'] = round_1_cohort.apply(
    lambda x: True if x.hc1disescn3 == ' 1 YES' else False if x.hc1disescn3 == ' 2 NO' else np.nan, axis=1)

# arthritis
# hc1disescn4
# 12 na
round_1_cohort['arthritis'] = round_1_cohort.apply(
    lambda x: True if x.hc1disescn4 == ' 1 YES' else False if x.hc1disescn4 == ' 2 NO' else np.nan, axis=1)

# diabetes
# hc1disescn6
# 2 na
round_1_cohort['diabetes'] = round_1_cohort.apply(
    lambda x: True if x.hc1disescn6 == ' 1 YES' else False if x.hc1disescn6 == ' 2 NO' else np.nan, axis=1)

# lung_disease
# hc1disescn7
# 4 na
round_1_cohort['lung_disease'] = round_1_cohort.apply(
    lambda x: True if x.hc1disescn7 == ' 1 YES' else False if x.hc1disescn7 == ' 2 NO' else np.nan, axis=1)

# stroke
# hc1disescn8
# 5 na
round_1_cohort['stroke'] = round_1_cohort.apply(
    lambda x: True if x.hc1disescn8 == ' 1 YES' else False if x.hc1disescn8 == ' 2 NO' else np.nan, axis=1)

# cancer
# hc1disescn10
# 2 na
round_1_cohort['cancer'] = round_1_cohort.apply(
    lambda x: True if x.hc1disescn10 == ' 1 YES' else False if x.hc1disescn10 == ' 2 NO' else np.nan, axis=1)

# Age category
# r1d2intvrage
# no na
# appended as age_category


def age_category(x):
    d = {
        '1 - 65-69': '65-69',
        '2 - 70-74': '70-74',
        '3 - 75-79': '75-79',
        '4 - 80-84': '80-84',
        '5 - 85-89': '85+',
        '6 - 90 +': '85+'
    }
    return d.get(x.r1d2intvrage, np.nan)


round_1_cohort['age_category'] = round_1_cohort.apply(age_category, axis=1)

# Obesity (defined by BMI)
# BMI >= 30 kg/m^2
# appended as Obesity
# no na due to exclusion criteria

round_1_cohort['obesity'] = round_1_cohort.apply(
    lambda x: True if x.bmi >= 30 else False, axis=1)

# Sarcopenic obesity definitions

# Grouping 1: sarcopenia, obesity, sarcopenic obesity, neither
# obesity derived from BMI (variable obesity)
# appended as grouping_1_so_status


def grouping_1_so_status(x):

    if pd.isna(x.sarcopenia) or pd.isna(x.obesity):  # shouldn't happen
        return np.nan

    if x.sarcopenia and not x.obesity:
        return 'Sarcopenia'
    elif not x.sarcopenia and x.obesity:
        return 'Obesity'
    elif x.sarcopenia and x.obesity:
        return 'Sarcopenic Obesity'
    elif not x.sarcopenia and not x.obesity:
        return 'Neither'


round_1_cohort['grouping_1_so_status'] = round_1_cohort.apply(
    grouping_1_so_status, axis=1)


def grouping_1_so_status_cutoff2(x):

    if pd.isna(x.sarcopenia_cutoff2) or pd.isna(x.obesity):  # shouldn't happen
        return np.nan

    if x.sarcopenia_cutoff2 and not x.obesity:
        return 'Sarcopenia'
    elif not x.sarcopenia_cutoff2 and x.obesity:
        return 'Obesity'
    elif x.sarcopenia_cutoff2 and x.obesity:
        return 'Sarcopenic Obesity'
    elif not x.sarcopenia_cutoff2 and not x.obesity:
        return 'Neither'


round_1_cohort['grouping_1_so_status_cutoff2'] = round_1_cohort.apply(
    grouping_1_so_status_cutoff2, axis=1)

# Grouping 2: sarcopenia, obesity, sarcopenic obesity, neither
# obesity derived from waist circumference (variable name high_wc)
# appended as grouping_2_so_status
# 104 na (due to missing wc)


def grouping_2_so_status(x):

    if pd.isna(x.sarcopenia) or pd.isna(x.high_wc):
        return np.nan

    if x.sarcopenia and not x.high_wc:
        return 'Sarcopenia'
    elif not x.sarcopenia and x.high_wc:
        return 'Obesity'
    elif x.sarcopenia and x.high_wc:
        return 'Sarcopenic Obesity'
    elif not x.sarcopenia and not x.high_wc:
        return 'Neither'


round_1_cohort['grouping_2_so_status'] = round_1_cohort.apply(
    grouping_2_so_status, axis=1)


def grouping_2_so_status_cutoff2(x):

    if pd.isna(x.sarcopenia_cutoff2) or pd.isna(x.high_wc):
        return np.nan

    if x.sarcopenia_cutoff2 and not x.high_wc:
        return 'Sarcopenia'
    elif not x.sarcopenia_cutoff2 and x.high_wc:
        return 'Obesity'
    elif x.sarcopenia_cutoff2 and x.high_wc:
        return 'Sarcopenic Obesity'
    elif not x.sarcopenia_cutoff2 and not x.high_wc:
        return 'Neither'


round_1_cohort['grouping_2_so_status_cutoff2'] = round_1_cohort.apply(
    grouping_2_so_status_cutoff2, axis=1)


# STEADI score
# Adapted from Dr. Matthew Lohman's SAS code

# Giving round option but requires refactoring if used for non baseline rounds
def STEADI_score(x, round):
    # screen 1 conditions
    if all([m in [' 2 NO', np.nan] for m in
            [x[f'hc{round}faleninyr'], x[f'hc{round}worryfall'], x[f'ss{round}prbbalcrd']]]):
        screen1 = 'Low risk'
    elif any([m == ' 1 YES' for m in
              [x[f'hc{round}faleninyr'], x[f'hc{round}worryfall'], x[f'ss{round}prbbalcrd']]]):
        screen1 = 'Medium risk'

    # screen 2 conditions
    screen2 = np.nan
    if screen1 == 'Medium risk':
        if x[f'ba{round}ftdmreslt'] == '2 Attempted, not held for 10 seconds':
            screen2 = 'Potential high risk'
        elif x[f'ba{round}ftdmreslt'] == '3 Not attempted' and any([a == ' 1 Yes' for a in [x[f'ba{round}rsn{i}ftstd'] for i in range(1, 4)]]):
            screen2 = 'Potential high risk'
        elif pd.isna(x[f'ba{round}ftdmreslt']) and x[f'ba{round}dblftadm'] in ['3 Not administered because did not complete prior balance tests', '4 Not Eligible due to exclusion criteria']:
            screen2 = 'Potential high risk'

        # space weirdness from pandas stata read-in
        if x[f'ch{round}2chstrslt'] == '2 Attempted    ':
            screen2 = 'Potential high risk'
        elif x[f'ch{round}2chstrslt'] == '3 Not attempted' and any([a == ' 1 Yes' for a in [x[f'ch{round}chstntat{i}'] for i in range(1, 4)]]):
            screen2 = 'Potential high risk'
        elif pd.isna(x[f'ch{round}2chstrslt']) and x[f'ch{round}drchradm'] in ['3 Not administered because did not complete single chair stand w/o arms', '4 Not Eligible due to exclusion criteria']:
            screen2 = 'Potential high risk'

    # screen 3 conditions
    if screen2 == 'Potential high risk' and (x[f'hc{round}multifall'] == ' 1 YES' or x[f'hc{round}brokebon1'] == ' 1 YES'):
        screen3 = 'High risk'
    else:
        screen3 = np.nan

    # STEADI
    if screen1 == 'Low risk':
        STEADI = 'Low risk'
    elif screen1 == 'Medium risk':
        if pd.isna(screen2) and pd.isna(screen3):
            STEADI = 'Medium risk'
        elif screen2 == 'Potential high risk' and pd.isna(screen3):
            STEADI = 'Medium risk'
        elif screen3 == 'High risk':
            STEADI = 'High risk'

    return STEADI


round_1_cohort['STEADI_score'] = round_1_cohort.apply(
    STEADI_score, round=1, axis=1)

round_1_cohort.to_csv('output_files/baseline_processed_2.23.2020.csv')
round_1_cohort.to_pickle('output_files/baseline_processed_2.23.2020.pkl')
