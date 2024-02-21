!pip3 install wikimapper

from sqlalchemy import create_engine

engine = create_engine("sqlite://" + path)

query = """
SELECT * 
  FROM mapping;
"""

weather = pd.read_sql(query, engine)


from wikimapper import WikiMapper
import os

path = '/content/drive/MyDrive/index_enwiki-latest.db'
mapper = WikiMapper(path)
wikidata_id = mapper.title_to_id("Python_(programming_language)")
print(wikidata_id) # Q28865

! cp index_fawiki-latest.db drive/MyDrive/.
! cp index_enwiki-latest.db drive/MyDrive/.