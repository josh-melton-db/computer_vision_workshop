# Databricks notebook source
def __validate_libraries():
    import requests
    try:
        site = "https://github.com/databricks-academy/dbacademy"
        response = requests.get(site)
        error = f"Unable to access GitHub or PyPi resources (HTTP {response.status_code} for {site})."
        assert response.status_code == 200, "{error} Please see the \"Troubleshooting | {section}\" section of the \"Version Info\" notebook for more information.".format(error=error, section="Cannot Install Libraries")
    except Exception as e:
        if type(e) is AssertionError: raise e
        error = f"Unable to access GitHub or PyPi resources ({site})."
        raise AssertionError("{error} Please see the \"Troubleshooting | {section}\" section of the \"Version Info\" notebook for more information.".format(error=error, section="Cannot Install Libraries")) from e

def __install_libraries():
    global pip_command
    
    specified_version = f"v3.0.70"
    key = "dbacademy.library.version"
    version = spark.conf.get(key, specified_version)

    if specified_version != version:
        print("** Dependency Version Overridden *******************************************************************")
        print(f"* This course was built for {specified_version} of the DBAcademy Library, but it is being overridden via the Spark")
        print(f"* configuration variable \"{key}\". The use of version v3.0.70 is not advised as we")
        print(f"* cannot guarantee compatibility with this version of the course.")
        print("****************************************************************************************************")

    try:
        from dbacademy import dbgems  
        installed_version = dbgems.lookup_current_module_version("dbacademy")
        if installed_version == version:
            pip_command = "list --quiet"  # Skipping pip install of pre-installed python library
        else:
            print(f"WARNING: The wrong version of dbacademy is attached to this cluster. Expected {version}, found {installed_version}.")
            print(f"Installing the correct version.")
            raise Exception("Forcing re-install")

    except Exception as e:
        # The import fails if library is not attached to cluster
        if not version.startswith("v"): library_url = f"git+https://github.com/databricks-academy/dbacademy@{version}"
        else: library_url = f"https://github.com/databricks-academy/dbacademy/releases/download/{version}/dbacademy-{version[1:]}-py3-none-any.whl"

        default_command = f"install --quiet --disable-pip-version-check {library_url}"
        pip_command = spark.conf.get("dbacademy.library.install", default_command)

        if pip_command != default_command:
            print(f"WARNING: Using alternative library installation:\n| default: %pip {default_command}\n| current: %pip {pip_command}")
        else:
            # We are using the default libraries; next we need to verify that we can reach those libraries.
            __validate_libraries()

__install_libraries()

# COMMAND ----------

# MAGIC %pip $pip_command

# COMMAND ----------

# MAGIC %run ./_dataset_index

# COMMAND ----------

import pyspark.sql.functions as F
from dbacademy import dbgems
from dbacademy.dbhelper import DBAcademyHelper, Paths, CourseConfig, LessonConfig

# The following attributes are externalized to make them easy
# for content developers to update with every new course.

course_config = CourseConfig(course_code = "dl",
                             course_name = "deep-learning-with-databricks",
                             data_source_name = "deep-learning-with-databricks",
                             data_source_version = "v03",
                             install_min_time = "3 min",
                             install_max_time = "10 min",
                             remote_files = remote_files,
                             supported_dbrs = ["12.2.x-cpu-ml-scala2.12"],
                             expected_dbrs = "12.2.x-cpu-ml-scala2.12")

# Defined here for the majority of lessons, 
# and later modified on a per-lesson basis.
lesson_config = LessonConfig(name = None,
                             create_schema = True,
                             create_catalog = False,
                             requires_uc = False,
                             installing_datasets = True,
                             enable_streaming_support = False,
                             enable_ml_support = True)

