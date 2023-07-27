get_ipython().run_line_magic('load_ext', 'sql')

get_ipython().run_line_magic('sql', 'sqlite://')

get_ipython().run_cell_magic('sql', '', "CREATE TABLE writer (first_name, last_name, year_of_death);\nINSERT INTO writer VALUES ('William', 'Shakespear', 1616);\nINSERT INTO writer VALUES ('Bertold', 'Brecht', 1956);")

get_ipython().run_line_magic('sql', 'select * from writer')



