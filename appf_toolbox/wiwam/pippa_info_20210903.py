import pandas as pd
import psycopg2
import psycopg2.extras
import pymysql
from psycopg2 import sql

from pick import pick


def pippa_info():

    lt_user = "readonlyuser"
    lt_password = "readonlyuser"
    lt_host = "tpa-plantdb.plantphenomics.org.au"
    lt_db = 'LTSystem_Production'
    lt_query = """
    SELECT NAME
    FROM LTDBS
    WHERE NAME like '0000_Production_S%'
        AND SUBSTRING(NAME, '0000_Production_S_(\d+)')::int > 2017
    ORDER BY NAME DESC;
    """

    lt_connection = psycopg2.connect(host=lt_host,
                                     user=lt_user,
                                     password=lt_password,
                                     dbname=lt_db,
                                     cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        with lt_connection.cursor() as cursor:
            cursor.execute(lt_query)
            old_prod_databases = cursor.fetchall()

    finally:
        lt_connection.close()

    databases = ['0000_Production_S'] + [x['name'] for x in old_prod_databases]

    measurement_labels = list()

    for database in databases:

        lt_query = """
        SELECT DISTINCT measurement_label, {} AS lt_db FROM snapshot
        JOIN tiled_image ON tiled_image.snapshot_id = snapshot.id
        WHERE camera_label = 'WIWAM';
        """

        lt_connection = psycopg2.connect(host=lt_host,
                                         user=lt_user,
                                         password=lt_password,
                                         dbname=database,
                                         cursor_factory=psycopg2.extras.RealDictCursor)
        try:
            with lt_connection.cursor() as cursor:
                cursor.execute(sql.SQL(lt_query)
                               .format(sql.Literal(database)))
                measurement_labels.extend(cursor.fetchall())

        finally:
            lt_connection.close()

    measurement_labels_df = pd.DataFrame(measurement_labels)

    option , index = pick(measurement_labels_df['measurement_label'].tolist(), "Select measurement_label")
    measurement_label_selection = measurement_labels_df.loc[measurement_labels_df['measurement_label'] == option]

    measurement_label = measurement_label_selection.iloc[0]['measurement_label']
    lt_db = measurement_label_selection.iloc[0]['lt_db']

    pippa_user = "pippa"
    pippa_password = "2308432JFMFWEsdfaswefa"
    pippa_host = "tpa-pippa.plantphenomics.org.au"
    pippa_db = "pippa"

    pippa_query = "SELECT value, image_url FROM pippa_pot_data JOIN pippa_image ON pippa_image.pot_id = pippa_pot_data.pot_id;"

    pippa_connection = pymysql.connect(host=pippa_host,
                                       user=pippa_user,
                                       password=pippa_password,
                                       db=pippa_db,
                                       # charset='utf8mb4',
                                       cursorclass=pymysql.cursors.DictCursor)

    try:
        with pippa_connection.cursor() as cursor:
            # Read a single record
            cursor.execute(pippa_query)
            pippa_data = cursor.fetchall()
            # print(pippa_data)
    finally:
        pippa_connection.close()

    lt_query = """
    SELECT * FROM (SELECT DISTINCT ON (snapshot.id_tag) snapshot.id_tag AS snapshot_id_tag 
    FROM snapshot WHERE measurement_label = %(ml)s) sub
    LEFT JOIN metadata_view ON sub.snapshot_id_tag = metadata_view.id_tag;
    """
    lt_query_params = {"ml": measurement_label}

    lt_connection = psycopg2.connect(host=lt_host,
                                     user=lt_user,
                                     password=lt_password,
                                     dbname=lt_db,
                                     cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        with lt_connection.cursor() as cursor:
            cursor.execute(lt_query, lt_query_params)
            lt_data = cursor.fetchall()
    finally:
        lt_connection.close()

    pippa_df = pd.DataFrame(pippa_data)
    lt_df = pd.DataFrame(lt_data)

    merged = lt_df.merge(pippa_df, left_on="snapshot_id_tag", right_on="value").drop_duplicates()

    print(merged)

    merged.to_csv('{}_merged.csv'.format(measurement_label), index=False)


if __name__ == "__main__":
    pippa_info()
