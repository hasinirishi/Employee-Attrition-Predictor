import pandas as pd
import numpy as np
from datetime import datetime

# Define standard features and their aliases
STANDARD_ALIASES = {
    "Age": ["age", "employeeage", "workerage", "currentage", "dob", "birthdate", "dateofbirth"],
    "BusinessTravel": ["businesstravel", "travel", "travelfrequency", "travelstatus"],
    "DailyRate": ["dailyrate", "dailyrate", "dayrate", "daily pay"],
    "Department": ["department", "dept", "departmentname", "team", "businessunit"],
    "DistanceFromHome": ["distancefromhome", "distance", "commute", "distfromhome", "commutedistance"],
    "Education": ["education", "educationlevel", "edu", "qualification"],
    "EducationField": ["educationfield", "field", "studyfield", "major", "specialization"],
    "EnvironmentSatisfaction": ["environmentsatisfaction", "envsatisfaction", "workenvironmentsatisfaction", "workplaceconditions"],
    "Gender": ["gender", "sex"],
    "HourlyRate": ["hourlyrate", "hourrate", "hourlypay"],
    "JobInvolvement": ["jobinvolvement", "involvement", "engagement", "jobengagement"],
    "JobLevel": ["joblevel", "grade", "level", "salaryband", "paygrade"],
    "JobRole": ["jobrole", "role", "designation", "position", "jobtitle"],
    "JobSatisfaction": ["jobsatisfaction", "employeesurvey", "satisfactionscore", "engagementscore", "happiness", "employeehappiness"],
    "MaritalStatus": ["maritalstatus", "marital", "status", "relationshipstatus"],
    "MonthlyIncome": ["monthlyincome", "salary", "income", "monthlysalary", "pay", "ctc", "annualincome", "yearlyincome", "annualsalary"],
    "MonthlyRate": ["monthlyrate", "monthrate"],
    "NumCompaniesWorked": ["numcompaniesworked", "companiesworked", "previouscompanies", "numbercompaniesworked"],
    "OverTime": ["overtime", "extrahours", "ot", "overtimehours"],
    "PercentSalaryHike": ["percentsalaryhike", "salaryhike", "hikepercent", "percenthike", "salaryraise"],
    "PerformanceRating": ["performancerating", "performance", "rating", "performancescore"],
    "RelationshipSatisfaction": ["relationshipsatisfaction", "relationsatisfaction", "peersatisfaction"],
    "StockOptionLevel": ["stockoptionlevel", "stockoptions", "equity", "stocklevel"],
    "TotalWorkingYears": ["totalworkingyears", "totalexperience", "experience", "workingyears"],
    "TrainingTimesLastYear": ["trainingtimeslastyear", "trainingtimes", "trainings", "traininghours"],
    "WorkLifeBalance": ["worklifebalance", "wlb", "worklifescore", "lifebalance"],
    "YearsAtCompany": ["yearsatcompany", "tenure", "yearsworked", "joindate", "hiredate", "datejoined", "companytenure"],
    "YearsInCurrentRole": ["yearsincurrentrole", "roleexperience", "yearsrole", "tenureinrole"],
    "YearsSinceLastPromotion": ["yearssincelastpromotion", "lastpromotion", "promotionyears", "yearssincepromotion"],
    "YearsWithCurrManager": ["yearswithcurrmanager", "manageryears", "yearsmanager", "tenurewithmanager"]
}

# Sub-aliases for deriving features (these maps help us figure out if we need to derive something)
DERIVATION_SOURCES = {
    "Age": ["dob", "birthdate", "dateofbirth", "birth_date"],
    "YearsAtCompany": ["joindate", "hiredate", "datejoined", "join_date", "hire_date"],
    "MonthlyIncome": ["annualincome", "yearlyincome", "annualsalary", "annual_income", "yearly_income", "annual_salary"]
}

INTELLIGENT_DEFAULTS = {
    "StockOptionLevel": 1,
    "JobLevel": 1,
    "BusinessTravel": "Travel_Rarely",
    "OverTime": "No",
    "EnvironmentSatisfaction": 3,
    "JobSatisfaction": 3,
    "WorkLifeBalance": 3,
    "PerformanceRating": 3,
    "JobInvolvement": 3,
    "Age": 35,
    "MonthlyIncome": 5000,
    "DistanceFromHome": 10,
    "YearsAtCompany": 5,
    "TotalWorkingYears": 8,
    "YearsInCurrentRole": 3,
    "YearsSinceLastPromotion": 1,
    "YearsWithCurrManager": 3,
    "DailyRate": 800,
    "HourlyRate": 65,
    "MonthlyRate": 15000,
    "NumCompaniesWorked": 1,
    "PercentSalaryHike": 12,
    "RelationshipSatisfaction": 3,
    "TrainingTimesLastYear": 2,
    "Education": 3,
    "Gender": "Male",
    "MaritalStatus": "Single",
    "EducationField": "Life Sciences",
    "Department": "Research & Development",
    "JobRole": "Sales Executive"
}

def clean_string(s):
    if not isinstance(s, str):
        return ""
    return "".join(c for c in s.lower() if c.isalnum())

def levenshtein_similarity(s1, s2):
    c1 = clean_string(s1)
    c2 = clean_string(s2)
    if not c1 or not c2:
        return 0.0
    if c1 == c2:
        return 1.0
    
    m, n = len(c1), len(c2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if c1[i - 1] == c2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + 1)
    distance = dp[m][n]
    max_len = max(m, n)
    return 1.0 - (distance / max_len)


class UniversalPreprocessor:
    def __init__(self, target_features=None):
        self.target_features = target_features if target_features is not None else list(STANDARD_ALIASES.keys())
        self.mappings = {}          # raw_column_name -> standard_feature_name
        self.derivations = {}       # standard_feature_name -> dict detailing derivation
        self.imputations = {}       # standard_feature_name -> method/value used
        self.ignored_columns = []
        self.unknown_columns = []
        self.report = {}
        
    def detect_schema(self, df):
        """Returns True if the dataframe matches the original IBM schema exactly."""
        clean_cols = set(df.columns)
        # Required columns for IBM (minus ignored or target Attrition, depending on the case)
        ibm_req = {"Age", "BusinessTravel", "DailyRate", "Department", "DistanceFromHome", "Education", 
                   "EducationField", "EnvironmentSatisfaction", "Gender", "HourlyRate", "JobInvolvement", 
                   "JobLevel", "JobRole", "JobSatisfaction", "MaritalStatus", "MonthlyIncome", "MonthlyRate", 
                   "NumCompaniesWorked", "OverTime", "PercentSalaryHike", "PerformanceRating", 
                   "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear", 
                   "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"}
        return ibm_req.issubset(clean_cols)

    def map_columns(self, df, user_overrides=None):
        """
        Phase 1: Build semantic column mapping.
        Iterate through dataframe columns and map to standard features.
        """
        self.mappings = {}
        self.unknown_columns = []
        self.ignored_columns = []
        
        user_overrides = user_overrides or {}
        
        # Add explicit overrides first
        for raw, target in user_overrides.items():
            if target == "Ignore":
                self.ignored_columns.append(raw)
            elif target in self.target_features:
                self.mappings[raw] = target
        
        # Map the remaining columns
        for col in df.columns:
            if col in self.mappings or col in self.ignored_columns or col in user_overrides:
                continue
            
            # Exact clean match
            cleaned_col = clean_string(col)
            best_match = None
            best_score = 0.0
            
            # Match against target features and their aliases
            for target_feat, aliases in STANDARD_ALIASES.items():
                if cleaned_col == clean_string(target_feat):
                    best_match = target_feat
                    best_score = 1.0
                    break
                for alias in aliases:
                    if cleaned_col == clean_string(alias):
                        best_match = target_feat
                        best_score = 1.0
                        break
            
            # If no exact match, try fuzzy matching
            if best_score < 1.0:
                for target_feat, aliases in STANDARD_ALIASES.items():
                    # Similarity to the main feature name
                    sim = levenshtein_similarity(col, target_feat)
                    if sim > best_score:
                        best_score = sim
                        best_match = target_feat
                    
                    # Similarity to aliases
                    for alias in aliases:
                        sim = levenshtein_similarity(col, alias)
                        if sim > best_score:
                            best_score = sim
                            best_match = target_feat
            
            # Auto-map threshold: similarity > 0.8
            if best_score >= 0.8 and best_match not in self.mappings.values():
                self.mappings[col] = best_match
            else:
                self.unknown_columns.append(col)
                
        # Fill in report
        self.report["mapped_columns"] = self.mappings.copy()
        self.report["unknown_columns"] = self.unknown_columns.copy()
        return self.mappings

    def suggest_llm_mapping(self, col):
        """
        Phase 6: Heuristic fallback simulating an LLM column understanding suggestion.
        """
        cleaned = clean_string(col)
        # Check standard definitions
        suggestions = {
            "happiness": ("JobSatisfaction", "Possibly JobSatisfaction"),
            "employeehappiness": ("JobSatisfaction", "Possibly JobSatisfaction"),
            "satisfactionscore": ("JobSatisfaction", "Satisfaction rating"),
            "salaryband": ("JobLevel", "Could relate to JobLevel or MonthlyIncome"),
            "paygrade": ("JobLevel", "Could relate to JobLevel or MonthlyIncome"),
            "grade": ("JobLevel", "Could relate to JobLevel"),
            "level": ("JobLevel", "Could relate to JobLevel"),
            "tenure": ("YearsAtCompany", "Could relate to YearsAtCompany"),
            "yearsworked": ("YearsAtCompany", "Could relate to YearsAtCompany"),
            "experience": ("TotalWorkingYears", "Could relate to TotalWorkingYears"),
            "worklifescore": ("WorkLifeBalance", "Could relate to WorkLifeBalance"),
            "engagement": ("JobInvolvement", "Could relate to JobInvolvement"),
            "extrahours": ("OverTime", "Could relate to OverTime"),
            "ot": ("OverTime", "Could relate to OverTime")
        }
        
        for kw, (target, desc) in suggestions.items():
            if kw in cleaned or levenshtein_similarity(col, kw) > 0.75:
                return target, desc
        return None, "No strong heuristic match found"

    def derive_features(self, df):
        """
        Phase 2: Derive missing important features.
        Modifies a copy of the dataframe to add derived features.
        """
        derived_df = pd.DataFrame(index=df.index)
        
        # Helper to find a raw column by searching clean mappings or raw name
        def find_raw_col(targets):
            for raw, mapped in self.mappings.items():
                if mapped in targets:
                    return raw
            for col in df.columns:
                if clean_string(col) in [clean_string(t) for t in targets]:
                    return col
            return None

        # 1. YearsAtCompany derivation
        mapped_years = [raw for raw, mapped in self.mappings.items() if mapped == "YearsAtCompany"]
        if not mapped_years:
            join_col = find_raw_col(DERIVATION_SOURCES["YearsAtCompany"])
            if join_col:
                try:
                    current_year = datetime.now().year
                    join_series = pd.to_datetime(df[join_col], errors="coerce")
                    derived_df["YearsAtCompany"] = join_series.apply(
                        lambda d: current_year - d.year if pd.notna(d) else np.nan
                    )
                    self.derivations["YearsAtCompany"] = {
                        "source": join_col,
                        "method": "CurrentYear - JoinYear"
                    }
                except Exception:
                    pass

        # 2. Age derivation
        mapped_age = [raw for raw, mapped in self.mappings.items() if mapped == "Age"]
        if not mapped_age:
            dob_col = find_raw_col(DERIVATION_SOURCES["Age"])
            if dob_col:
                try:
                    current_date = datetime.now()
                    dob_series = pd.to_datetime(df[dob_col], errors="coerce")
                    derived_df["Age"] = dob_series.apply(
                        lambda d: int((current_date - d).days / 365.25) if pd.notna(d) else np.nan
                    )
                    self.derivations["Age"] = {
                        "source": dob_col,
                        "method": "CurrentDate - DOB"
                    }
                except Exception:
                    pass

        # 3. MonthlyIncome derivation
        mapped_income = [raw for raw, mapped in self.mappings.items() if mapped == "MonthlyIncome"]
        if not mapped_income:
            annual_col = find_raw_col(DERIVATION_SOURCES["MonthlyIncome"])
            if annual_col:
                try:
                    income_series = pd.to_numeric(df[annual_col], errors="coerce")
                    # If values are extremely large, assume annual income
                    derived_df["MonthlyIncome"] = income_series / 12.0
                    self.derivations["MonthlyIncome"] = {
                        "source": annual_col,
                        "method": "AnnualIncome / 12"
                    }
                except Exception:
                    pass

        # 4. JobSatisfaction derivation
        mapped_satisfaction = [raw for raw, mapped in self.mappings.items() if mapped == "JobSatisfaction"]
        if not mapped_satisfaction:
            survey_col = find_raw_col(["employeesurvey", "satisfactionscore", "engagementscore", "happiness"])
            if survey_col:
                try:
                    score_series = pd.to_numeric(df[survey_col], errors="coerce")
                    # Normalize survey score to standard 1-4 scale if possible
                    max_val = score_series.max()
                    if pd.notna(max_val) and max_val > 4:
                        derived_df["JobSatisfaction"] = np.round((score_series / max_val) * 3 + 1)
                    else:
                        derived_df["JobSatisfaction"] = score_series
                    self.derivations["JobSatisfaction"] = {
                        "source": survey_col,
                        "method": "Scaled score from survey"
                    }
                except Exception:
                    pass

        return derived_df

    def handle_missing(self, df, training_medians=None):
        """
        Phase 3: Handle missing/absent features by using training median/mode or defaults.
        """
        imputed_df = df.copy()
        
        for feat in self.target_features:
            if feat not in imputed_df.columns or imputed_df[feat].isna().all():
                # Feature is completely missing
                if training_medians is not None and feat in training_medians:
                    fill_val = training_medians[feat]
                    imputed_df[feat] = fill_val
                    self.imputations[feat] = f"training median ({fill_val})"
                else:
                    fill_val = INTELLIGENT_DEFAULTS.get(feat, 0)
                    imputed_df[feat] = fill_val
                    self.imputations[feat] = f"intelligent default ({fill_val})"
            else:
                # Feature exists but has some missing values
                nulls = imputed_df[feat].isna()
                if nulls.any():
                    if training_medians is not None and feat in training_medians:
                        fill_val = training_medians[feat]
                        imputed_df[feat] = imputed_df[feat].fillna(fill_val)
                        self.imputations[feat] = f"training median for subset ({fill_val})"
                    else:
                        fill_val = INTELLIGENT_DEFAULTS.get(feat, 0)
                        imputed_df[feat] = imputed_df[feat].fillna(fill_val)
                        self.imputations[feat] = f"intelligent default for subset ({fill_val})"
                        
        return imputed_df

    def encode_categories(self, df, encoders):
        """
        Encode string and categorical columns using original label encoders,
        handling unseen categories gracefully.
        """
        encoded_df = df.copy()
        for col in encoded_df.columns:
            if col in encoders:
                le = encoders[col]
                # Cast to string if required
                vals = encoded_df[col].astype(str)
                allowed = set(le.classes_)
                
                # Check if it looks already encoded numeric
                try:
                    num_vals = pd.to_numeric(vals, errors="coerce")
                    if num_vals.notna().all() and num_vals.between(0, len(le.classes_) - 1).all():
                        encoded_df[col] = num_vals.astype(int)
                        continue
                except Exception:
                    pass
                
                # Treat unseen categories by mapping to the most frequent category or the first
                fallback = le.classes_[0]
                cleaned_vals = vals.apply(lambda x: x if x in allowed else fallback)
                encoded_df[col] = le.transform(cleaned_vals)
            else:
                # Ensure numeric
                encoded_df[col] = pd.to_numeric(encoded_df[col], errors="coerce").fillna(0)
        return encoded_df

    def align_to_model(self, df, feature_names):
        """
        Phase 4: Select only model features and ensure they are ordered exactly matching the model.
        """
        return df[feature_names].copy()

    def generate_report(self):
        """
        Compile report dictionary detailing mapping, derivations, and imputations.
        """
        return {
            "mapped": self.mappings,
            "derived": self.derivations,
            "imputed": self.imputations,
            "ignored": self.ignored_columns,
            "unknown": self.unknown_columns
        }

    def get_profile_summary(self, df):
        """
        Phase 5: Generate dataset summary, profile metrics, and Data Quality Score.
        """
        rows, cols = df.shape
        missing_count = df.isna().sum().sum()
        total_cells = rows * cols if rows * cols > 0 else 1
        missing_pct = (missing_count / total_cells) * 100
        
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        categorical_cols = list(df.select_dtypes(exclude=[np.number]).columns)
        
        mapped_count = len(self.mappings)
        derived_count = len(self.derivations)
        unknown_count = len(self.unknown_columns)
        
        # Calculate Data Quality Score (0 - 100)
        # Deduct score for missing variables, high percentage of missing cells, and unknown columns
        score = 100.0
        
        # Penalty for missing standard features (each missing counts -1.5)
        missing_feats_count = len(self.target_features) - mapped_count - derived_count
        score -= (missing_feats_count * 2.0)
        
        # Penalty for cell missingness
        score -= (missing_pct * 1.5)
        
        # Penalty for unmapped columns in custom dataset (only if not IBM format)
        is_ibm = self.detect_schema(df)
        if not is_ibm and cols > 0:
            score -= (unknown_count / cols) * 15.0
            
        score = max(0.0, min(100.0, score))
        
        # Construct suggestions
        suggestions = []
        if missing_feats_count > 0:
            suggestions.append(f"⚠️ {missing_feats_count} required model features are missing and will be imputed using defaults. Consider adding columns for: " + 
                               ", ".join([f for f in self.target_features if f not in self.mappings.values() and f not in self.derivations][:3]))
        if missing_pct > 5.0:
            suggestions.append(f"📉 The dataset has {missing_pct:.1f}% missing values. Ensure data completeness to reduce bias.")
        if unknown_count > 0:
            suggestions.append(f"🔍 {unknown_count} columns were not mapped. Use the column mapping utility to map them if they contain salary, age, or tenure data.")
        if score > 85:
            suggestions.append("✅ Excellent dataset quality! Ready for accurate predictions.")
        elif score > 60:
            suggestions.append("ℹ️ Good dataset quality, but some features are missing or need mapping.")
        else:
            suggestions.append("⚠️ Poor dataset quality. High risk of prediction errors due to heavy imputation.")
            
        return {
            "rows": rows,
            "columns": cols,
            "missing_pct": missing_pct,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "mapped_columns_count": mapped_count,
            "derived_columns_count": derived_count,
            "unknown_columns_count": unknown_count,
            "score": round(score),
            "suggestions": suggestions
        }
