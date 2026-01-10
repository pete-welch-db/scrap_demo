# Databricks notebook source
# MAGIC %md
# MAGIC # Scrap Data Generation - Bronze Layer
# MAGIC 
# MAGIC This notebook generates synthetic scrap data for an automotive Tier 1/2 supplier demo.
# MAGIC 
# MAGIC **Data Model:**
# MAGIC - Suppliers Master
# MAGIC - Plants/Facilities
# MAGIC - Work Centers
# MAGIC - Equipment/Machines
# MAGIC - Operators
# MAGIC - Cost Centers
# MAGIC - Defect Codes (Hierarchical)
# MAGIC - Parts Master
# MAGIC - Scrap Transactions (60K records, 3 years)
# MAGIC - 8D Quality Reports
# MAGIC 
# MAGIC **Target Schema:** `scrap_catalog.scrap_demo`

# COMMAND ----------

# MAGIC %pip install faker

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import random
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any
import numpy as np
from faker import Faker
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as F

# Initialize
fake = Faker()
Faker.seed(42)
random.seed(42)
np.random.seed(42)

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Configuration - Use widget parameters from job, with defaults for interactive use
dbutils.widgets.text("catalog", "scrap_catalog", "Catalog Name")
dbutils.widgets.text("schema", "scrap_demo", "Schema Name")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
FULL_SCHEMA = f"{CATALOG}.{SCHEMA}"

print(f"Using catalog: {CATALOG}, schema: {SCHEMA}")

# Data volume settings
NUM_PARTS = 500
NUM_SCRAP_RECORDS = 60000
HISTORY_YEARS = 3
NUM_SUPPLIERS = 45
NUM_PLANTS = 6
NUM_WORK_CENTERS_PER_PLANT = 8
NUM_EQUIPMENT_PER_WORK_CENTER = 5
NUM_OPERATORS_PER_PLANT = 50
NUM_COST_CENTERS = 20

# Date range
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=HISTORY_YEARS * 365)

# COMMAND ----------

try:
    # Create schema if not exists
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {FULL_SCHEMA}")
except:
    pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Suppliers Master

# COMMAND ----------

# Automotive supplier types and specializations
SUPPLIER_SPECIALIZATIONS = [
    "Stamping & Metal Forming",
    "Plastic Injection Molding",
    "Die Casting",
    "Machining & CNC",
    "Electrical Components",
    "Wire Harnesses",
    "Rubber & Sealing",
    "Glass & Glazing",
    "Interior Trim",
    "Fasteners & Hardware",
    "Coatings & Finishing",
    "Electronics & Sensors"
]

SUPPLIER_TIERS = ["Tier 1", "Tier 2", "Tier 3"]
SUPPLIER_REGIONS = ["US-Midwest", "US-Southeast", "US-Southwest", "Canada", "Mexico"]

# Some suppliers will have quality issues (for ML patterns)
PROBLEMATIC_SUPPLIER_RATIO = 0.15

def generate_suppliers(num_suppliers: int) -> List[Dict]:
    suppliers = []
    num_problematic = int(num_suppliers * PROBLEMATIC_SUPPLIER_RATIO)
    problematic_indices = set(random.sample(range(num_suppliers), num_problematic))
    
    for i in range(num_suppliers):
        supplier_id = f"SUP-{str(i+1).zfill(4)}"
        is_problematic = i in problematic_indices
        
        # Problematic suppliers have lower quality scores
        if is_problematic:
            quality_score = round(random.uniform(0.65, 0.82), 3)
            on_time_delivery = round(random.uniform(0.75, 0.88), 3)
        else:
            quality_score = round(random.uniform(0.92, 0.99), 3)
            on_time_delivery = round(random.uniform(0.90, 0.99), 3)
        
        suppliers.append({
            "supplier_id": supplier_id,
            "supplier_name": fake.company(),
            "supplier_tier": random.choices(SUPPLIER_TIERS, weights=[0.3, 0.5, 0.2])[0],
            "specialization": random.choice(SUPPLIER_SPECIALIZATIONS),
            "region": random.choice(SUPPLIER_REGIONS),
            "address": fake.address().replace("\n", ", "),
            "contact_name": fake.name(),
            "contact_email": fake.company_email(),
            "contact_phone": fake.phone_number(),
            "quality_score": quality_score,
            "on_time_delivery_rate": on_time_delivery,
            "is_preferred": not is_problematic and random.random() > 0.5,
            "contract_start_date": fake.date_between(start_date="-5y", end_date="-1y"),
            "contract_end_date": fake.date_between(start_date="+1y", end_date="+3y"),
            "is_problematic_flag": is_problematic,  # Hidden flag for ML validation
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        })
    
    return suppliers

suppliers_data = generate_suppliers(NUM_SUPPLIERS)
suppliers_df = spark.createDataFrame(suppliers_data)
suppliers_df.write.mode("overwrite").saveAsTable(f"{FULL_SCHEMA}.suppliers_bronze")
print(f"Created {NUM_SUPPLIERS} suppliers")
suppliers_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Plants/Facilities

# COMMAND ----------

PLANT_LOCATIONS = [
    {"city": "Detroit", "state": "MI", "country": "USA", "timezone": "America/Detroit"},
    {"city": "Louisville", "state": "KY", "country": "USA", "timezone": "America/Kentucky/Louisville"},
    {"city": "San Antonio", "state": "TX", "country": "USA", "timezone": "America/Chicago"},
    {"city": "Oakville", "state": "ON", "country": "Canada", "timezone": "America/Toronto"},
    {"city": "Saltillo", "state": "Coahuila", "country": "Mexico", "timezone": "America/Monterrey"},
    {"city": "Silao", "state": "Guanajuato", "country": "Mexico", "timezone": "America/Mexico_City"},
]

PLANT_TYPES = ["Assembly", "Stamping", "Powertrain", "Body Shop", "Paint Shop"]

def generate_plants() -> List[Dict]:
    plants = []
    for i, loc in enumerate(PLANT_LOCATIONS):
        plant_id = f"PLT-{str(i+1).zfill(3)}"
        
        # Some plants have older equipment (for ML patterns)
        avg_equipment_age = random.randint(3, 15) if i < 4 else random.randint(8, 20)
        
        plants.append({
            "plant_id": plant_id,
            "plant_name": f"{loc['city']} {random.choice(PLANT_TYPES)} Plant",
            "plant_code": f"{loc['city'][:3].upper()}{i+1}",
            "city": loc["city"],
            "state_province": loc["state"],
            "country": loc["country"],
            "timezone": loc["timezone"],
            "square_footage": random.randint(500000, 2000000),
            "year_opened": random.randint(1985, 2015),
            "avg_equipment_age_years": avg_equipment_age,
            "annual_capacity_units": random.randint(100000, 500000),
            "num_employees": random.randint(500, 3000),
            "is_active": True,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        })
    
    return plants

plants_data = generate_plants()
plants_df = spark.createDataFrame(plants_data)
plants_df.write.mode("overwrite").saveAsTable(f"{FULL_SCHEMA}.plants_bronze")
print(f"Created {len(plants_data)} plants")
plants_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Work Centers

# COMMAND ----------

WORK_CENTER_TYPES = [
    {"type": "Stamping Press", "defect_prone": True, "category": "Forming"},
    {"type": "Welding Cell", "defect_prone": True, "category": "Joining"},
    {"type": "CNC Machining", "defect_prone": False, "category": "Machining"},
    {"type": "Injection Molding", "defect_prone": True, "category": "Molding"},
    {"type": "Assembly Line", "defect_prone": False, "category": "Assembly"},
    {"type": "Paint Booth", "defect_prone": True, "category": "Finishing"},
    {"type": "Quality Inspection", "defect_prone": False, "category": "QC"},
    {"type": "Packaging", "defect_prone": False, "category": "Logistics"},
    {"type": "Heat Treatment", "defect_prone": True, "category": "Processing"},
    {"type": "Die Casting", "defect_prone": True, "category": "Casting"},
]

def generate_work_centers(plants: List[Dict]) -> List[Dict]:
    work_centers = []
    wc_counter = 1
    
    for plant in plants:
        # Each plant gets a subset of work center types
        plant_wc_types = random.sample(WORK_CENTER_TYPES, NUM_WORK_CENTERS_PER_PLANT)
        
        for wc_type in plant_wc_types:
            work_center_id = f"WC-{str(wc_counter).zfill(4)}"
            
            work_centers.append({
                "work_center_id": work_center_id,
                "work_center_name": f"{wc_type['type']} - {plant['plant_code']}",
                "work_center_type": wc_type["type"],
                "work_center_category": wc_type["category"],
                "plant_id": plant["plant_id"],
                "capacity_per_hour": random.randint(50, 500),
                "num_stations": random.randint(2, 12),
                "is_defect_prone": wc_type["defect_prone"],
                "maintenance_schedule": random.choice(["Weekly", "Bi-Weekly", "Monthly"]),
                "last_maintenance_date": fake.date_between(start_date="-60d", end_date="today"),
                "is_active": True,
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            })
            wc_counter += 1
    
    return work_centers

work_centers_data = generate_work_centers(plants_data)
work_centers_df = spark.createDataFrame(work_centers_data)
work_centers_df.write.mode("overwrite").saveAsTable(f"{FULL_SCHEMA}.work_centers_bronze")
print(f"Created {len(work_centers_data)} work centers")
work_centers_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Equipment/Machines

# COMMAND ----------

EQUIPMENT_TYPES = {
    "Stamping Press": ["Hydraulic Press", "Mechanical Press", "Servo Press", "Transfer Press"],
    "Welding Cell": ["Spot Welder", "MIG Welder", "Laser Welder", "Robotic Welder"],
    "CNC Machining": ["CNC Lathe", "CNC Mill", "5-Axis CNC", "Grinding Machine"],
    "Injection Molding": ["Injection Molder", "Blow Molder", "Compression Molder"],
    "Assembly Line": ["Conveyor System", "Robotic Arm", "Torque Station", "Vision System"],
    "Paint Booth": ["Spray Booth", "E-Coat System", "Curing Oven", "Paint Robot"],
    "Quality Inspection": ["CMM Machine", "Vision Inspection", "X-Ray System", "Leak Tester"],
    "Packaging": ["Packaging Robot", "Labeling Machine", "Palletizer"],
    "Heat Treatment": ["Furnace", "Quench Tank", "Tempering Oven"],
    "Die Casting": ["Die Cast Machine", "Trim Press", "Shot Blast"],
}

EQUIPMENT_MANUFACTURERS = ["Fanuc", "ABB", "KUKA", "Yaskawa", "Siemens", "Bosch", "Trumpf", "DMG Mori", "Mazak", "Haas"]

def generate_equipment(work_centers: List[Dict], plants: List[Dict]) -> List[Dict]:
    equipment = []
    eq_counter = 1
    
    # Get plant avg equipment age for correlation
    plant_age_map = {p["plant_id"]: p["avg_equipment_age_years"] for p in plants}
    
    for wc in work_centers:
        wc_type = wc["work_center_type"]
        eq_types = EQUIPMENT_TYPES.get(wc_type, ["Generic Equipment"])
        plant_avg_age = plant_age_map.get(wc["plant_id"], 10)
        
        for _ in range(random.randint(3, NUM_EQUIPMENT_PER_WORK_CENTER)):
            equipment_id = f"EQ-{str(eq_counter).zfill(5)}"
            
            # Equipment age correlates with plant age (for ML patterns)
            age_years = max(1, int(np.random.normal(plant_avg_age, 3)))
            install_date = datetime.now() - timedelta(days=age_years * 365)
            
            # Older equipment has more downtime and lower OEE
            if age_years > 12:
                oee = round(random.uniform(0.65, 0.80), 3)
                mtbf_hours = random.randint(200, 500)
            elif age_years > 7:
                oee = round(random.uniform(0.75, 0.88), 3)
                mtbf_hours = random.randint(400, 800)
            else:
                oee = round(random.uniform(0.85, 0.95), 3)
                mtbf_hours = random.randint(700, 1500)
            
            equipment.append({
                "equipment_id": equipment_id,
                "equipment_name": f"{random.choice(eq_types)} #{eq_counter}",
                "equipment_type": random.choice(eq_types),
                "manufacturer": random.choice(EQUIPMENT_MANUFACTURERS),
                "model_number": f"{fake.lexify(text='???').upper()}-{random.randint(1000, 9999)}",
                "serial_number": fake.uuid4()[:12].upper(),
                "work_center_id": wc["work_center_id"],
                "plant_id": wc["plant_id"],
                "installation_date": install_date.date(),
                "age_years": age_years,
                "oee_rating": oee,
                "mtbf_hours": mtbf_hours,
                "last_calibration_date": fake.date_between(start_date="-90d", end_date="today"),
                "next_maintenance_date": fake.date_between(start_date="today", end_date="+60d"),
                "status": random.choices(["Operational", "Under Maintenance", "Idle"], weights=[0.85, 0.10, 0.05])[0],
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            })
            eq_counter += 1
    
    return equipment

equipment_data = generate_equipment(work_centers_data, plants_data)
equipment_df = spark.createDataFrame(equipment_data)
equipment_df.write.mode("overwrite").saveAsTable(f"{FULL_SCHEMA}.equipment_bronze")
print(f"Created {len(equipment_data)} equipment records")
equipment_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Operators

# COMMAND ----------

SHIFTS = [1, 2, 3]
SKILL_LEVELS = ["Trainee", "Junior", "Intermediate", "Senior", "Expert"]

def generate_operators(plants: List[Dict]) -> List[Dict]:
    operators = []
    op_counter = 1
    
    for plant in plants:
        for _ in range(NUM_OPERATORS_PER_PLANT):
            operator_id = f"OP-{str(op_counter).zfill(5)}"
            
            years_experience = random.randint(0, 25)
            
            # Skill level correlates with experience
            if years_experience < 1:
                skill_level = "Trainee"
            elif years_experience < 3:
                skill_level = "Junior"
            elif years_experience < 7:
                skill_level = "Intermediate"
            elif years_experience < 15:
                skill_level = "Senior"
            else:
                skill_level = "Expert"
            
            # Some operators are more error-prone (for ML patterns)
            is_high_error = random.random() < 0.1
            
            operators.append({
                "operator_id": operator_id,
                "employee_number": f"EMP{random.randint(10000, 99999)}",
                "first_name": fake.first_name(),
                "last_name": fake.last_name(),
                "plant_id": plant["plant_id"],
                "primary_shift": random.choice(SHIFTS),
                "skill_level": skill_level,
                "years_experience": years_experience,
                "certifications": random.randint(1, 8),
                "hire_date": fake.date_between(start_date=f"-{years_experience+1}y", end_date=f"-{years_experience}y") if years_experience > 0 else fake.date_between(start_date="-1y", end_date="today"),
                "is_active": random.random() > 0.05,
                "is_high_error_flag": is_high_error,  # Hidden for ML validation
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            })
            op_counter += 1
    
    return operators

operators_data = generate_operators(plants_data)
operators_df = spark.createDataFrame(operators_data)
operators_df.write.mode("overwrite").saveAsTable(f"{FULL_SCHEMA}.operators_bronze")
print(f"Created {len(operators_data)} operators")
operators_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Cost Centers

# COMMAND ----------

COST_CENTER_TYPES = [
    "Direct Labor", "Indirect Labor", "Raw Materials", "WIP Inventory",
    "Finished Goods", "Quality Control", "Maintenance", "Engineering",
    "Tooling", "Utilities", "Depreciation", "Scrap & Rework",
    "Overtime", "Training", "Safety", "Environmental",
    "Packaging", "Shipping", "Administrative", "R&D"
]

def generate_cost_centers() -> List[Dict]:
    cost_centers = []
    
    for i, cc_type in enumerate(COST_CENTER_TYPES[:NUM_COST_CENTERS]):
        cost_center_id = f"CC-{str(i+1).zfill(3)}"
        
        cost_centers.append({
            "cost_center_id": cost_center_id,
            "cost_center_name": cc_type,
            "cost_center_code": f"{cc_type[:3].upper()}{i+1:02d}",
            "department": random.choice(["Operations", "Quality", "Engineering", "Finance", "Logistics"]),
            "budget_annual": round(random.uniform(100000, 5000000), 2),
            "manager_name": fake.name(),
            "is_active": True,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        })
    
    return cost_centers

cost_centers_data = generate_cost_centers()
cost_centers_df = spark.createDataFrame(cost_centers_data)
cost_centers_df.write.mode("overwrite").saveAsTable(f"{FULL_SCHEMA}.cost_centers_bronze")
print(f"Created {len(cost_centers_data)} cost centers")
cost_centers_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Defect Codes (Hierarchical)

# COMMAND ----------

DEFECT_HIERARCHY = {
    "DIM": {
        "name": "Dimensional",
        "subcodes": [
            ("DIM-001", "Out of Tolerance - Length", 0.15),
            ("DIM-002", "Out of Tolerance - Width", 0.12),
            ("DIM-003", "Out of Tolerance - Diameter", 0.10),
            ("DIM-004", "Out of Tolerance - Thickness", 0.08),
            ("DIM-005", "Warpage/Distortion", 0.18),
            ("DIM-006", "Hole Location Error", 0.07),
        ]
    },
    "SRF": {
        "name": "Surface Defects",
        "subcodes": [
            ("SRF-001", "Scratches", 0.20),
            ("SRF-002", "Dents/Dings", 0.15),
            ("SRF-003", "Porosity", 0.12),
            ("SRF-004", "Orange Peel", 0.08),
            ("SRF-005", "Paint Defects", 0.10),
            ("SRF-006", "Corrosion/Rust", 0.05),
        ]
    },
    "MAT": {
        "name": "Material Defects",
        "subcodes": [
            ("MAT-001", "Wrong Material Grade", 0.05),
            ("MAT-002", "Material Contamination", 0.08),
            ("MAT-003", "Inclusions", 0.10),
            ("MAT-004", "Hardness Out of Spec", 0.07),
            ("MAT-005", "Chemical Composition", 0.04),
        ]
    },
    "ASM": {
        "name": "Assembly Defects",
        "subcodes": [
            ("ASM-001", "Missing Component", 0.12),
            ("ASM-002", "Wrong Component", 0.08),
            ("ASM-003", "Improper Torque", 0.10),
            ("ASM-004", "Misalignment", 0.15),
            ("ASM-005", "Incomplete Assembly", 0.06),
        ]
    },
    "WLD": {
        "name": "Welding Defects",
        "subcodes": [
            ("WLD-001", "Incomplete Fusion", 0.12),
            ("WLD-002", "Porosity in Weld", 0.10),
            ("WLD-003", "Crack in Weld", 0.08),
            ("WLD-004", "Spatter", 0.15),
            ("WLD-005", "Undercut", 0.07),
        ]
    },
    "HDL": {
        "name": "Handling Damage",
        "subcodes": [
            ("HDL-001", "Shipping Damage", 0.10),
            ("HDL-002", "Storage Damage", 0.08),
            ("HDL-003", "Fork Lift Damage", 0.05),
            ("HDL-004", "Packaging Failure", 0.07),
        ]
    },
    "EQP": {
        "name": "Equipment Related",
        "subcodes": [
            ("EQP-001", "Tool Wear", 0.20),
            ("EQP-002", "Machine Malfunction", 0.15),
            ("EQP-003", "Calibration Drift", 0.10),
            ("EQP-004", "Die/Mold Damage", 0.12),
        ]
    },
    "PRC": {
        "name": "Process Related",
        "subcodes": [
            ("PRC-001", "Temperature Out of Range", 0.12),
            ("PRC-002", "Pressure Out of Range", 0.10),
            ("PRC-003", "Cycle Time Deviation", 0.08),
            ("PRC-004", "Improper Setup", 0.15),
        ]
    },
}

def generate_defect_codes() -> List[Dict]:
    defect_codes = []
    
    for category_code, category_data in DEFECT_HIERARCHY.items():
        # Add category level
        defect_codes.append({
            "defect_code": category_code,
            "defect_name": category_data["name"],
            "defect_category": category_data["name"],
            "parent_code": None,
            "hierarchy_level": 1,
            "severity_weight": round(random.uniform(0.5, 1.0), 2),
            "is_active": True,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        })
        
        # Add subcodes
        for subcode, subname, freq in category_data["subcodes"]:
            defect_codes.append({
                "defect_code": subcode,
                "defect_name": subname,
                "defect_category": category_data["name"],
                "parent_code": category_code,
                "hierarchy_level": 2,
                "severity_weight": round(random.uniform(0.3, 0.9), 2),
                "is_active": True,
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            })
    
    return defect_codes

defect_codes_data = generate_defect_codes()
defect_codes_df = spark.createDataFrame(defect_codes_data)
defect_codes_df.write.mode("overwrite").saveAsTable(f"{FULL_SCHEMA}.defect_codes_bronze")
print(f"Created {len(defect_codes_data)} defect codes")
defect_codes_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Parts Master

# COMMAND ----------

PART_CATEGORIES = {
    "Body Panels": ["Hood", "Fender", "Door Panel", "Roof Panel", "Trunk Lid", "Quarter Panel", "Bumper Cover"],
    "Structural": ["A-Pillar", "B-Pillar", "Rocker Panel", "Cross Member", "Subframe", "Reinforcement"],
    "Powertrain": ["Engine Block", "Cylinder Head", "Intake Manifold", "Exhaust Manifold", "Transmission Case", "Flywheel"],
    "Chassis": ["Control Arm", "Knuckle", "Brake Rotor", "Brake Caliper", "Suspension Link", "Stabilizer Bar"],
    "Interior": ["Dashboard", "Door Trim", "Console", "Headliner", "Seat Frame", "Armrest"],
    "Electrical": ["Wire Harness", "Connector", "Sensor Housing", "ECU Bracket", "Fuse Box", "Battery Tray"],
    "Fasteners": ["Bolt Assembly", "Nut Plate", "Clip", "Bracket", "Retainer", "Spacer"],
}

MATERIALS = ["Steel", "Aluminum", "Plastic", "Composite", "Rubber", "Magnesium", "Cast Iron"]
UNIT_OF_MEASURE = ["EA", "KG", "M", "SET", "PAIR"]

def generate_parts(num_parts: int, suppliers: List[Dict]) -> List[Dict]:
    parts = []
    supplier_ids = [s["supplier_id"] for s in suppliers]
    problematic_supplier_ids = [s["supplier_id"] for s in suppliers if s["is_problematic_flag"]]
    
    for i in range(num_parts):
        part_id = f"PRT-{str(i+1).zfill(5)}"
        category = random.choice(list(PART_CATEGORIES.keys()))
        part_type = random.choice(PART_CATEGORIES[category])
        
        # Assign supplier - some parts from problematic suppliers
        if random.random() < 0.2 and problematic_supplier_ids:
            supplier_id = random.choice(problematic_supplier_ids)
        else:
            supplier_id = random.choice(supplier_ids)
        
        # Cost varies by category
        if category in ["Powertrain", "Chassis"]:
            unit_cost = round(random.uniform(50, 500), 2)
        elif category in ["Body Panels", "Structural"]:
            unit_cost = round(random.uniform(20, 200), 2)
        elif category == "Electrical":
            unit_cost = round(random.uniform(10, 150), 2)
        else:
            unit_cost = round(random.uniform(5, 100), 2)
        
        # Critical parts have higher scrap impact
        is_critical = random.random() < 0.15
        
        parts.append({
            "part_id": part_id,
            "part_number": f"{category[:3].upper()}-{random.randint(10000, 99999)}-{fake.lexify(text='??').upper()}",
            "part_name": f"{part_type} - {fake.word().title()}",
            "part_description": f"{part_type} for automotive application. {fake.sentence()}",
            "part_category": category,
            "part_type": part_type,
            "material": random.choice(MATERIALS),
            "unit_of_measure": random.choice(UNIT_OF_MEASURE),
            "unit_cost": unit_cost,
            "unit_weight_kg": round(random.uniform(0.1, 25), 3),
            "supplier_id": supplier_id,
            "lead_time_days": random.randint(3, 45),
            "safety_stock": random.randint(50, 500),
            "is_critical": is_critical,
            "engineering_revision": f"Rev {random.choice(['A', 'B', 'C', 'D', 'E'])}",
            "effective_date": fake.date_between(start_date="-3y", end_date="-6m"),
            "is_active": True,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        })
    
    return parts

parts_data = generate_parts(NUM_PARTS, suppliers_data)
parts_df = spark.createDataFrame(parts_data)
parts_df.write.mode("overwrite").saveAsTable(f"{FULL_SCHEMA}.parts_master_bronze")
print(f"Created {NUM_PARTS} parts")
parts_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Scrap Transactions (60K records with ML patterns)

# COMMAND ----------

def get_defect_weights_for_work_center(wc_type: str) -> Dict[str, float]:
    """Return weighted defect codes based on work center type"""
    base_weights = {code: 0.1 for code in [d["defect_code"] for d in defect_codes_data if d["hierarchy_level"] == 2]}
    
    # Adjust weights based on work center type
    if "Stamping" in wc_type:
        for code in base_weights:
            if code.startswith("DIM") or code.startswith("SRF"):
                base_weights[code] *= 3.0
    elif "Welding" in wc_type:
        for code in base_weights:
            if code.startswith("WLD"):
                base_weights[code] *= 5.0
    elif "Paint" in wc_type:
        for code in base_weights:
            if code.startswith("SRF"):
                base_weights[code] *= 4.0
    elif "Assembly" in wc_type:
        for code in base_weights:
            if code.startswith("ASM"):
                base_weights[code] *= 4.0
    elif "Machining" in wc_type or "CNC" in wc_type:
        for code in base_weights:
            if code.startswith("DIM") or code.startswith("EQP"):
                base_weights[code] *= 3.0
    
    return base_weights

def generate_scrap_transactions(
    num_records: int,
    parts: List[Dict],
    work_centers: List[Dict],
    equipment: List[Dict],
    operators: List[Dict],
    suppliers: List[Dict],
    defect_codes: List[Dict],
    cost_centers: List[Dict]
) -> List[Dict]:
    
    transactions = []
    
    # Build lookup maps
    part_supplier_map = {p["part_id"]: p["supplier_id"] for p in parts}
    supplier_quality_map = {s["supplier_id"]: s["quality_score"] for s in suppliers}
    problematic_suppliers = {s["supplier_id"] for s in suppliers if s["is_problematic_flag"]}
    
    wc_plant_map = {wc["work_center_id"]: wc["plant_id"] for wc in work_centers}
    wc_type_map = {wc["work_center_id"]: wc["work_center_type"] for wc in work_centers}
    
    equipment_by_wc = {}
    for eq in equipment:
        wc_id = eq["work_center_id"]
        if wc_id not in equipment_by_wc:
            equipment_by_wc[wc_id] = []
        equipment_by_wc[wc_id].append(eq)
    
    old_equipment = {eq["equipment_id"] for eq in equipment if eq["age_years"] > 12}
    
    operators_by_plant = {}
    for op in operators:
        plant_id = op["plant_id"]
        if plant_id not in operators_by_plant:
            operators_by_plant[plant_id] = []
        operators_by_plant[plant_id].append(op)
    
    high_error_operators = {op["operator_id"] for op in operators if op["is_high_error_flag"]}
    
    leaf_defects = [d for d in defect_codes if d["hierarchy_level"] == 2]
    scrap_cost_center = next((cc for cc in cost_centers if "Scrap" in cc["cost_center_name"]), cost_centers[0])
    
    # Generate dates with patterns
    all_dates = []
    current_date = START_DATE
    while current_date <= END_DATE:
        all_dates.append(current_date)
        current_date += timedelta(days=1)
    
    for i in range(num_records):
        transaction_id = f"SCR-{str(i+1).zfill(7)}"
        
        # Select work center and related entities
        wc = random.choice(work_centers)
        wc_id = wc["work_center_id"]
        plant_id = wc_plant_map[wc_id]
        wc_type = wc_type_map[wc_id]
        
        # Select equipment from work center
        wc_equipment = equipment_by_wc.get(wc_id, [])
        eq = random.choice(wc_equipment) if wc_equipment else None
        equipment_id = eq["equipment_id"] if eq else None
        
        # Select operator from plant
        plant_operators = operators_by_plant.get(plant_id, [])
        op = random.choice(plant_operators) if plant_operators else None
        operator_id = op["operator_id"] if op else None
        
        # Select part
        part = random.choice(parts)
        part_id = part["part_id"]
        supplier_id = part_supplier_map[part_id]
        
        # Select date with seasonal patterns
        base_date = random.choice(all_dates)
        
        # SEASONAL PATTERNS
        month = base_date.month
        day_of_week = base_date.weekday()
        
        # Higher scrap in summer (model year changeover) and winter (holiday skeleton crew)
        seasonal_multiplier = 1.0
        if month in [7, 8]:  # Summer shutdown period
            seasonal_multiplier = 1.4
        elif month in [12, 1]:  # Winter/holiday
            seasonal_multiplier = 1.25
        elif month in [3, 4]:  # Spring ramp-up
            seasonal_multiplier = 1.15
        
        # Monday effect (startup issues)
        if day_of_week == 0:
            seasonal_multiplier *= 1.2
        
        # Apply seasonal pattern to quantity
        base_qty = random.randint(1, 20)
        quantity = int(base_qty * seasonal_multiplier)
        quantity = max(1, min(quantity, 50))  # Cap at reasonable values
        
        # SHIFT PATTERNS - 3rd shift has more issues
        shift = random.choices([1, 2, 3], weights=[0.4, 0.35, 0.25])[0]
        if shift == 3:
            quantity = int(quantity * 1.15)
        
        # SUPPLIER QUALITY CORRELATION
        if supplier_id in problematic_suppliers:
            quantity = int(quantity * random.uniform(1.2, 1.5))
        
        # EQUIPMENT AGE CORRELATION
        if equipment_id and equipment_id in old_equipment:
            quantity = int(quantity * random.uniform(1.1, 1.3))
        
        # OPERATOR CORRELATION
        if operator_id and operator_id in high_error_operators:
            quantity = int(quantity * random.uniform(1.15, 1.35))
        
        quantity = max(1, quantity)
        
        # Select defect code weighted by work center type
        defect_weights = get_defect_weights_for_work_center(wc_type)
        defect = random.choices(
            leaf_defects,
            weights=[defect_weights.get(d["defect_code"], 0.1) for d in leaf_defects]
        )[0]
        
        # Calculate scrap cost
        unit_cost = part["unit_cost"]
        total_cost = round(unit_cost * quantity, 2)
        
        # Generate timestamp
        hour = random.randint(6, 22)
        minute = random.randint(0, 59)
        transaction_datetime = base_date.replace(hour=hour, minute=minute, second=random.randint(0, 59))
        
        transactions.append({
            "scrap_transaction_id": transaction_id,
            "transaction_datetime": transaction_datetime,
            "transaction_date": base_date.date(),
            "shift": shift,
            "part_id": part_id,
            "part_number": part["part_number"],
            "quantity_scrapped": quantity,
            "unit_cost": unit_cost,
            "total_scrap_cost": total_cost,
            "defect_code": defect["defect_code"],
            "defect_category": defect["defect_category"],
            "work_center_id": wc_id,
            "equipment_id": equipment_id,
            "operator_id": operator_id,
            "plant_id": plant_id,
            "supplier_id": supplier_id,
            "cost_center_id": scrap_cost_center["cost_center_id"],
            "lot_number": f"LOT-{fake.bothify(text='??###??').upper()}",
            "work_order": f"WO-{random.randint(100000, 999999)}",
            "disposition": random.choices(
                ["Scrap", "Rework", "Salvage", "Return to Supplier"],
                weights=[0.6, 0.25, 0.10, 0.05]
            )[0],
            "root_cause_category": random.choice(["Machine", "Method", "Material", "Man", "Measurement", "Environment"]),
            "inspector_notes": fake.sentence() if random.random() > 0.7 else None,
            "is_containment_required": random.random() < 0.15,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        })
        
        if (i + 1) % 10000 == 0:
            print(f"Generated {i + 1} scrap transactions...")
    
    return transactions

print("Generating scrap transactions with ML patterns...")
scrap_data = generate_scrap_transactions(
    NUM_SCRAP_RECORDS,
    parts_data,
    work_centers_data,
    equipment_data,
    operators_data,
    suppliers_data,
    defect_codes_data,
    cost_centers_data
)

scrap_df = spark.createDataFrame(scrap_data)
scrap_df.write.mode("overwrite").saveAsTable(f"{FULL_SCHEMA}.scrap_transactions_bronze")
print(f"Created {NUM_SCRAP_RECORDS} scrap transactions")
scrap_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. 8D Quality Reports

# COMMAND ----------

# Root cause templates for realistic text generation
ROOT_CAUSE_TEMPLATES = [
    "Investigation revealed that {defect_type} was caused by {cause}. The {factor} was found to be operating outside of specification, leading to {outcome}. Further analysis using {method} confirmed that {root_cause}.",
    "5-Why analysis determined that the primary root cause was {root_cause}. Contributing factors included {factor} and {secondary_factor}. The {equipment} showed signs of {condition}, which directly impacted {process}.",
    "Fishbone diagram analysis identified {root_cause} as the main contributor. The {defect_type} originated from {source} due to {cause}. Process capability study (Cpk) showed {metric} was below acceptable limits.",
    "Root cause investigation using 8D methodology revealed that {root_cause}. The failure mode was traced to {factor}, which occurred during {process}. Pareto analysis showed that {percentage}% of defects were attributable to this single cause.",
    "Statistical process control (SPC) data indicated that {process} went out of control on {timeframe}. The special cause variation was attributed to {root_cause}. Gage R&R study confirmed measurement system was {status}.",
]

CAUSES = [
    "worn tooling exceeding replacement interval",
    "temperature fluctuation in the process zone", 
    "contaminated raw material from supplier batch",
    "improper machine setup after changeover",
    "calibration drift in measurement equipment",
    "operator training gap on new procedure",
    "maintenance schedule not followed",
    "incorrect process parameters loaded",
    "environmental humidity out of specification",
    "supplier material property variation",
    "die wear beyond acceptable tolerance",
    "hydraulic pressure fluctuation",
    "fixture misalignment",
    "coolant concentration out of range",
    "programming error in CNC code",
]

CORRECTIVE_ACTION_TEMPLATES = [
    "Immediate containment: {containment}. Short-term corrective action: {short_term}. Long-term corrective action: {long_term}. Verification method: {verification}. Responsible: {owner}. Target completion: {date}.",
    "D3 Containment: {containment}. D5 Permanent Corrective Action: {long_term}. D6 Validation: {verification} performed with {method}. D7 Prevention: {prevention}. Lessons learned documented in {system}.",
    "Corrective actions implemented: 1) {short_term} - completed {date}. 2) {long_term} - in progress. 3) {prevention} - scheduled. Effectiveness verified through {verification}. PPAP update required: {ppap_status}.",
    "Action plan: {short_term} implemented immediately. Engineering change {ec_number} initiated for {long_term}. Control plan updated to include {control}. Operator training completed for {training_count} personnel.",
    "Resolution summary: Root cause eliminated through {long_term}. Process FMEA updated with new failure mode (RPN reduced from {rpn_before} to {rpn_after}). Statistical validation shows Cpk improved to {cpk}. Customer notified per {customer_protocol}.",
]

CONTAINMENT_ACTIONS = [
    "100% inspection of suspect inventory",
    "quarantine of affected lot numbers",
    "sorting of WIP and finished goods",
    "temporary process parameter adjustment",
    "additional quality gate established",
    "rework station setup for affected parts",
]

SHORT_TERM_ACTIONS = [
    "tooling replacement completed",
    "process parameters adjusted and locked",
    "equipment recalibrated",
    "operator retrained on procedure",
    "temporary inspection point added",
    "supplier notified and corrective action requested",
]

LONG_TERM_ACTIONS = [
    "preventive maintenance schedule revised to weekly interval",
    "new tooling design with improved wear characteristics implemented",
    "automated in-process inspection system installed",
    "supplier quality agreement updated with tighter specifications",
    "error-proofing (poka-yoke) device designed and installed",
    "process capability improvement project initiated",
    "engineering change to part design reducing sensitivity",
    "new supplier qualification completed as backup source",
]

PREVENTION_ACTIONS = [
    "lessons learned shared across all plants",
    "design FMEA updated with new failure mode",
    "standardized work instructions revised",
    "training program updated for all shifts",
    "similar processes audited for same risk",
]

def generate_8d_text(defect_category: str, is_detailed: bool = True) -> tuple:
    """Generate realistic 8D root cause and corrective action text"""
    
    root_cause_template = random.choice(ROOT_CAUSE_TEMPLATES)
    root_cause_text = root_cause_template.format(
        defect_type=defect_category.lower() + " defect",
        cause=random.choice(CAUSES),
        factor=random.choice(["primary equipment", "incoming material", "process parameter", "environmental condition"]),
        secondary_factor=random.choice(["shift handover communication", "documentation gap", "training deficiency"]),
        outcome=random.choice(["dimensional variation", "surface quality degradation", "assembly interference", "functional failure"]),
        method=random.choice(["5-Why analysis", "Ishikawa diagram", "fault tree analysis", "DOE"]),
        root_cause=random.choice(CAUSES),
        equipment=random.choice(["stamping press", "welding robot", "CNC machine", "injection molder"]),
        condition=random.choice(["excessive wear", "calibration drift", "mechanical looseness", "thermal expansion"]),
        process=random.choice(["forming operation", "welding sequence", "machining cycle", "assembly process"]),
        source=random.choice(["upstream operation", "material handling", "supplier shipment", "storage conditions"]),
        metric=random.choice(["Cpk", "Ppk", "process mean"]),
        percentage=random.randint(60, 85),
        timeframe=random.choice(["the second shift", "Monday startup", "after maintenance", "during changeover"]),
        status=random.choice(["adequate", "marginal", "requiring improvement"]),
    )
    
    corrective_template = random.choice(CORRECTIVE_ACTION_TEMPLATES)
    corrective_text = corrective_template.format(
        containment=random.choice(CONTAINMENT_ACTIONS),
        short_term=random.choice(SHORT_TERM_ACTIONS),
        long_term=random.choice(LONG_TERM_ACTIONS),
        prevention=random.choice(PREVENTION_ACTIONS),
        verification=random.choice(["dimensional inspection", "visual audit", "functional test", "SPC monitoring"]),
        method=random.choice(["Gage R&R", "capability study", "first article inspection", "process audit"]),
        owner=fake.name(),
        date=fake.date_between(start_date="-30d", end_date="+30d").strftime("%Y-%m-%d"),
        system=random.choice(["quality management system", "lessons learned database", "PFMEA library"]),
        ppap_status=random.choice(["Yes - Level 3", "No", "Yes - Level 1"]),
        ec_number=f"EC-{random.randint(10000, 99999)}",
        control=random.choice(["additional measurement point", "increased sampling frequency", "automated inspection"]),
        training_count=random.randint(5, 25),
        rpn_before=random.randint(150, 400),
        rpn_after=random.randint(20, 100),
        cpk=round(random.uniform(1.33, 2.0), 2),
        customer_protocol=random.choice(["PRR process", "customer portal submission", "quality alert system"]),
    )
    
    return root_cause_text, corrective_text


def generate_8d_reports(scrap_transactions: List[Dict], parts: List[Dict]) -> List[Dict]:
    """Generate 8D reports - one 8D can cover multiple scrap records"""
    
    reports = []
    part_map = {p["part_id"]: p for p in parts}
    
    # Group scrap transactions by defect code and time window to create 8Ds
    # Sort by date first
    sorted_transactions = sorted(scrap_transactions, key=lambda x: x["transaction_date"])
    
    # Create 8D groupings - each 8D covers scrap records within a week with same defect category
    eight_d_groups = {}
    
    for txn in sorted_transactions:
        # Create grouping key: defect category + week + plant
        week_num = txn["transaction_date"].isocalendar()[1]
        year = txn["transaction_date"].year
        group_key = f"{txn['defect_category']}_{year}_{week_num}_{txn['plant_id']}"
        
        if group_key not in eight_d_groups:
            eight_d_groups[group_key] = []
        eight_d_groups[group_key].append(txn)
    
    report_counter = 1
    scrap_to_8d_map = {}
    
    for group_key, txns in eight_d_groups.items():
        eight_d_id = f"8D-{str(report_counter).zfill(6)}"
        
        # Get unique parts in this 8D
        unique_parts = list(set([t["part_id"] for t in txns]))
        primary_part = txns[0]["part_id"]
        part_info = part_map.get(primary_part, {})
        
        # Total quantities and costs
        total_qty = sum(t["quantity_scrapped"] for t in txns)
        total_cost = sum(t["total_scrap_cost"] for t in txns)
        
        defect_category = txns[0]["defect_category"]
        
        # Generate realistic text
        root_cause_text, corrective_action_text = generate_8d_text(defect_category)
        
        # Determine severity based on cost and quantity
        if total_cost > 10000 or total_qty > 100:
            severity = "Critical"
        elif total_cost > 5000 or total_qty > 50:
            severity = "Major"
        elif total_cost > 1000 or total_qty > 20:
            severity = "Moderate"
        else:
            severity = "Minor"
        
        # Status based on date
        first_date = min(t["transaction_date"] for t in txns)
        days_old = (datetime.now().date() - first_date).days
        
        if days_old < 14:
            status = "Open"
        elif days_old < 30:
            status = random.choices(["Open", "In Progress", "Pending Verification"], weights=[0.2, 0.5, 0.3])[0]
        elif days_old < 60:
            status = random.choices(["In Progress", "Pending Verification", "Closed"], weights=[0.2, 0.3, 0.5])[0]
        else:
            status = random.choices(["Pending Verification", "Closed"], weights=[0.2, 0.8])[0]
        
        # Problem description
        problem_description = f"Multiple {defect_category.lower()} defects detected affecting {len(unique_parts)} part number(s). " \
                            f"Total of {total_qty} units scrapped with estimated cost impact of ${total_cost:,.2f}. " \
                            f"Defects identified during {random.choice(['in-process inspection', 'final inspection', 'customer audit', 'internal audit'])}. " \
                            f"{fake.sentence()}"
        
        reports.append({
            "eight_d_id": eight_d_id,
            "report_number": f"8D-{year}-{str(report_counter).zfill(4)}",
            "title": f"{defect_category} Issue - {part_info.get('part_type', 'Multiple Parts')}",
            "status": status,
            "severity": severity,
            "priority": random.choice(["P1", "P2", "P3"]) if severity in ["Critical", "Major"] else random.choice(["P3", "P4"]),
            
            # D1 - Team
            "team_lead": fake.name(),
            "team_members": ", ".join([fake.name() for _ in range(random.randint(3, 7))]),
            
            # D2 - Problem Description
            "problem_description": problem_description,
            "defect_category": defect_category,
            "primary_defect_code": txns[0]["defect_code"],
            
            # Parts affected
            "primary_part_id": primary_part,
            "affected_part_ids": ",".join(unique_parts[:10]),  # Limit to first 10
            "num_parts_affected": len(unique_parts),
            
            # Quantities and costs
            "total_quantity_affected": total_qty,
            "total_scrap_cost": round(total_cost, 2),
            "num_scrap_records": len(txns),
            
            # Location info
            "plant_id": txns[0]["plant_id"],
            "work_center_id": txns[0]["work_center_id"],
            "supplier_id": txns[0]["supplier_id"],
            
            # D3 - Containment
            "containment_action": random.choice(CONTAINMENT_ACTIONS) + ". " + fake.sentence(),
            "containment_date": (first_date + timedelta(days=random.randint(1, 3))),
            
            # D4 - Root Cause Analysis  
            "root_cause_analysis": root_cause_text,
            "root_cause_category": random.choice(["Machine", "Method", "Material", "Man", "Measurement", "Environment"]),
            "is_escape_point_identified": random.random() > 0.3,
            
            # D5 - Corrective Actions
            "corrective_actions": corrective_action_text,
            
            # D6 - Verification
            "verification_method": random.choice(["Capability Study", "Inspection Data Review", "Audit", "Customer Feedback", "SPC Analysis"]),
            "verification_date": (first_date + timedelta(days=random.randint(14, 45))) if status in ["Pending Verification", "Closed"] else None,
            "verification_result": random.choice(["Effective", "Partially Effective", "Requires Additional Action"]) if status == "Closed" else None,
            
            # D7 - Prevention
            "preventive_actions": random.choice(PREVENTION_ACTIONS) + ". " + fake.sentence(),
            
            # D8 - Recognition
            "lessons_learned": f"Key learning: {fake.sentence()} Recommended for horizontal deployment to {random.choice(['all plants', 'similar processes', 'supplier base'])}.",
            
            # Dates
            "date_opened": first_date,
            "date_closed": (first_date + timedelta(days=random.randint(30, 90))) if status == "Closed" else None,
            "target_close_date": first_date + timedelta(days=random.randint(30, 60)),
            
            # Customer info
            "customer_complaint_number": f"CCR-{random.randint(10000, 99999)}" if random.random() < 0.3 else None,
            "is_customer_notified": random.random() < 0.4,
            
            # Metadata
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        })
        
        # Map scrap transactions to this 8D
        for txn in txns:
            scrap_to_8d_map[txn["scrap_transaction_id"]] = eight_d_id
        
        report_counter += 1
    
    return reports, scrap_to_8d_map

print("Generating 8D reports...")
eight_d_data, scrap_8d_mapping = generate_8d_reports(scrap_data, parts_data)

eight_d_df = spark.createDataFrame(eight_d_data)
eight_d_df.write.mode("overwrite").saveAsTable(f"{FULL_SCHEMA}.eight_d_reports_bronze")
print(f"Created {len(eight_d_data)} 8D reports covering {NUM_SCRAP_RECORDS} scrap transactions")
eight_d_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Scrap to 8D Mapping Table

# COMMAND ----------

# Create the mapping table
mapping_data = [
    {
        "scrap_transaction_id": scrap_id,
        "eight_d_id": eight_d_id,
        "created_at": datetime.now()
    }
    for scrap_id, eight_d_id in scrap_8d_mapping.items()
]

mapping_df = spark.createDataFrame(mapping_data)
mapping_df.write.mode("overwrite").saveAsTable(f"{FULL_SCHEMA}.scrap_8d_mapping_bronze")
print(f"Created mapping table with {len(mapping_data)} records")
mapping_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Validation & Summary

# COMMAND ----------

# Summary statistics
print("=" * 60)
print("BRONZE LAYER DATA GENERATION COMPLETE")
print("=" * 60)

tables = [
    ("suppliers_bronze", NUM_SUPPLIERS),
    ("plants_bronze", NUM_PLANTS),
    ("work_centers_bronze", NUM_PLANTS * NUM_WORK_CENTERS_PER_PLANT),
    ("equipment_bronze", "Variable"),
    ("operators_bronze", NUM_PLANTS * NUM_OPERATORS_PER_PLANT),
    ("cost_centers_bronze", NUM_COST_CENTERS),
    ("defect_codes_bronze", "Variable"),
    ("parts_master_bronze", NUM_PARTS),
    ("scrap_transactions_bronze", NUM_SCRAP_RECORDS),
    ("eight_d_reports_bronze", "Variable"),
    ("scrap_8d_mapping_bronze", NUM_SCRAP_RECORDS),
]

print(f"\nSchema: {FULL_SCHEMA}")
print("-" * 60)

for table_name, expected in tables:
    count = spark.table(f"{FULL_SCHEMA}.{table_name}").count()
    print(f"{table_name}: {count:,} records")

print("-" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ML Pattern Validation

# COMMAND ----------

# Validate embedded patterns for ML
print("\n=== ML PATTERN VALIDATION ===\n")

# 1. Seasonal pattern
print("1. SEASONAL SCRAP PATTERN:")
seasonal_df = spark.sql(f"""
    SELECT 
        month(transaction_date) as month,
        round(avg(quantity_scrapped), 2) as avg_qty,
        count(*) as num_records
    FROM {FULL_SCHEMA}.scrap_transactions_bronze
    GROUP BY month(transaction_date)
    ORDER BY month
""")
seasonal_df.display()

# 2. Shift pattern  
print("\n2. SHIFT-BASED PATTERN:")
shift_df = spark.sql(f"""
    SELECT 
        shift,
        round(avg(quantity_scrapped), 2) as avg_qty,
        round(sum(total_scrap_cost), 2) as total_cost,
        count(*) as num_records
    FROM {FULL_SCHEMA}.scrap_transactions_bronze
    GROUP BY shift
    ORDER BY shift
""")
shift_df.display()

# 3. Supplier quality correlation
print("\n3. SUPPLIER QUALITY CORRELATION:")
supplier_df = spark.sql(f"""
    SELECT 
        s.is_problematic_flag,
        count(*) as num_scrap_records,
        round(avg(t.quantity_scrapped), 2) as avg_qty_scrapped,
        round(sum(t.total_scrap_cost), 2) as total_scrap_cost
    FROM {FULL_SCHEMA}.scrap_transactions_bronze t
    JOIN {FULL_SCHEMA}.suppliers_bronze s ON t.supplier_id = s.supplier_id
    GROUP BY s.is_problematic_flag
""")
supplier_df.display()

# 4. Equipment age correlation
print("\n4. EQUIPMENT AGE CORRELATION:")
equipment_df = spark.sql(f"""
    SELECT 
        CASE 
            WHEN e.age_years > 12 THEN 'Old (>12 years)'
            WHEN e.age_years > 7 THEN 'Medium (7-12 years)'
            ELSE 'New (<7 years)'
        END as equipment_age_group,
        count(*) as num_scrap_records,
        round(avg(t.quantity_scrapped), 2) as avg_qty_scrapped
    FROM {FULL_SCHEMA}.scrap_transactions_bronze t
    JOIN {FULL_SCHEMA}.equipment_bronze e ON t.equipment_id = e.equipment_id
    GROUP BY CASE 
            WHEN e.age_years > 12 THEN 'Old (>12 years)'
            WHEN e.age_years > 7 THEN 'Medium (7-12 years)'
            ELSE 'New (<7 years)'
        END
""")
equipment_df.display()

print("\n=== BRONZE LAYER GENERATION COMPLETE ===")

