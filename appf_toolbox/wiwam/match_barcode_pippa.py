def match_barcode_pippa():
    """
    Match barcode of the wiwam system to pippa data. 
    """
    import pymysql
    import psycopg2, psycopg2.extras
    import pandas as pd

    import sys

    lt_user = "readonlyuser"
    lt_password = "readonlyuser"
    lt_host = "tpa-plantdb.plantphenomics.org.au"
    lt_db = 'LTSystem_Production'
    lt_query = "SELECT name FROM ltdbs;"

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

    lt_db = 'LTSystem'
    lt_connection = psycopg2.connect(host=lt_host,
                                     user=lt_user,
                                     password=lt_password,
                                     dbname=lt_db,
                                     cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        with lt_connection.cursor() as cursor:
            cursor.execute(lt_query)
            prod_databases = cursor.fetchall()

    finally:
        lt_connection.close()

    databases = [x['name'] for x in prod_databases] + [x['name'] for x in old_prod_databases]

    for i in range(0, len(databases)):
        print("{}:\t{}".format(i, databases[i]))

    d = int(input("Select Database: "))

    lt_query = "SELECT DISTINCT measurement_label FROM snapshot ORDER by measurement_label;"
    lt_db = databases[d]

    lt_connection = psycopg2.connect(host=lt_host,
                                     user=lt_user,
                                     password=lt_password,
                                     dbname=lt_db,
                                     cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        with lt_connection.cursor() as cursor:
            cursor.execute(lt_query)
            measurement_labels = cursor.fetchall()

    finally:
        lt_connection.close()

    for i in range(0, len(measurement_labels)):
        print("{}:\t{}".format(i, measurement_labels[i]['measurement_label']))

    selection = int(input("Enter a number: "))

    measurement_label = measurement_labels[selection]['measurement_label']

    pippa_user = "pippa"
    pippa_password = "2308432JFMFWEsdfaswefa"
    pippa_host = "tpa-pippa.plantphenomics.org.au"
    pippa_db = "pippa"

    pippa_query = "select value, image_url from pippa_pot_data join pippa_image on pippa_image.pot_id = pippa_pot_data.pot_id;"

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

    lt_query = """select * from (select distinct on (snapshot.id_tag) snapshot.id_tag as snapshot_id_tag from snapshot where measurement_label = %(ml)s) sub
                left join metadata_view on sub.snapshot_id_tag = metadata_view.id_tag
                ;"""
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
