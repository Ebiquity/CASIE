#!/usr/bin/env python
# coding: utf-8



"""

wd_search(string, required_types=[], limit=10) Search for up to 10
cyber-relevant entities whose label or alias matches string
and has at least one type in required_types, if provided.  candidates found are filtered by their immediate and inherited types to require that they have at least one cybersecurity-relevant type (e.g., malware) and no typoes from a blacklist (e.g., musical artist).

The return value is a list, roughly ordered from best to worst match.  For example, searching for 'wannacry' produces two hits:

 [{ 'concepturi':'http://www.wikidata.org/entity/Q29957041',
    'description':'ransomware cyberattack',
    'id':'Q29957041',
    'label':'WannaCry ransomware attack',
    'match':{ 'language':'en', 'text':'WannaCry ransomware attack', 'type':'label'},
    'types':[('Q4071928','cyberattack')]},
  { 'concepturi':'http://www.wikidata.org/entity/Q29908721',
    'description':'Ransomware',
    'id':'Q29908721',
    'label':'WannaCry',
    'match':{'language':'en', 'text':'WannaCry', 'type':'label'},
    'types':[('Q14001','malware'), ('Q7397','software')]}]

You can call this from the command line for experimentation, e.g. usage:

   python3 wd_search.py <string> [<required types>]
   python3 wd_search.py Adobe
   python3 wd_search.py python Q7397
   python3 wd_search.py mitre  "Q783794,Q2659904"

"""

from SPARQLWrapper import SPARQLWrapper, JSON
import pywikibot
import argparse
import pprint


# make wikidata the default for pywikibot search
default_search_site = pywikibot.Site("wikidata", "wikidata")


# sparql endpoints
wd_endpoint = "https://query.wikidata.org/bigdata/namespace/wdq/sparql"
dbpedia = "http://dbpedia.org/sparql"
default_endpoint = wd_endpoint


# find items with the given string as a label or alias
q_label_alias = """select ?item ?itemLabel ?type ?typeLabel where {{
   ?item rdfs:label|skos:altLabel "{STR}"@en; wdt:P31/wdt:P279* ?type .
   SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}}}"""

# get item types and their labels
q_types_labels = """
select ?type ?typeLabel where {{
   wd:{QID} (wdt:P31/wdt:P279*)|wdt:P279+ ?type .
   SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}}} 
"""

# get item types
q_types = """select ?type where {{ wd:{QID} wdt:P31/wdt:P279* ?type .}}"""


# searching wikidata returns entities some of whch are of interest and
# others not.  We filter these by requiring an entitie to have a type
# on the whitelist and no types on the backlist

# cybersecurity-relevant wikidata types: one of these must be a type
# of a good search result

wd_whitelist = {
  'Q5': 'human',
  'Q43229': 'organization',
  'Q82794': 'geographic region',
  'Q1048835': 'political territorial entity',
  'Q7397': 'software',
  'Q205663': 'process',
  'Q68': 'computer',
  'Q1301371': 'network',
  'Q14001': 'malware',
  'Q783794': 'company',
  'Q161157': 'password',
  'Q1541645': 'process identifier',
  'Q4418000': 'network address',
  'Q5830907': 'computer memory',
  'Q82753': 'computer file',
  'Q2904148': 'information leak',
  'Q4071928': 'cyberattack',
  'Q477202': 'cryptographic hash function',
  'Q141090': 'encryption',
  'Q5227362': 'data theft',
  'Q631425': 'computer vulnerability',
  'Q627226': 'Common Vulnerabilities and Exposures',
  'Q2801262': 'hacker group',
  'Q2798820': 'security hacker',
  'Q8142': 'currency',
  'Q2587068': 'sensitive information',
  'Q3966': 'computer hardware',
  'Q17517': 'mobile phone',
  'Q986008': 'payment system',
  'Q13479982': 'cryptocurrency',
  'Q20826013': 'software version',
  'Q20631656': 'software release',
  'Q44601380': 'property that may violate privacy',
  'Q1058914': 'software company',
  'Q278610': 'Dropper',
  'Q1332289':'black hat',
  'Q22685':'hacktivism',
  'Q47913':'intelligence agency',
  'Q28344495':'computer security consultant',
  'Q26102':'whistleblower',
  'Q317671':'botnet',
  'Q9135':'operating system',
  'Q4825885':'authentication protocol',
  'Q2659904':'government organization',
  'Q1668024':'service on internet',
  'Q202833':'social media'
}

# wikidata types that should not be in a search result
wd_blacklist = {
  'Q4438121':'sports organization',
  'Q11410':'game',
  'Q14897293':'fictional entity',
  'Q32178211':'music organisation',
  'Q16010345':'performer',
  'Q483501':'artist',
  'Q56678558':'unknown composer author',
  'Q28555911':'ordinary matter',
  'Q49848':'document'
}

# set of wikidata types for entities of interest
wd_whitelist_types = set(wd_whitelist.keys())

# set of wikidata types for entities not of interest
wd_blacklist_types = set(wd_blacklist.keys())


def prettyPrint(item):
    pprint.PrettyPrinter(indent=2).pprint(item)
    
def wd_entity_id(url):
    """ returns entity id if url is an entity, else the url"""
    return url.split('http://www.wikidata.org/entity/')[1] if url.startswith('http://www.wikidata.org/entity/') else url



def wd_search(string, required_types=[], limit=10):
    """ search for up to limit cyber-relevant entities whose label or
        alias matches string and has a t least one type in
        required_types, if not [] """
    candidates = wd_name_search(string, limit=limit)
    hits = []
    # get each item's types
    for item in candidates['search']:
        types = get_types(item['id'], required_types=required_types)
        if types:
            item['types'] = list(types)
            # remmove unwanted properties
            item.pop('repository', None)
            item.pop('url', None)
            item.pop('pageid', None)
            item.pop('title', None)
            hits.append(item)
    return hits

def wd_name_search(name, site=default_search_site, limit=10):
    # search wikidata for entities whose label of alias matches name
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102'}
    params = {'action':'wbsearchentities', 'format':'json', 'language':'en','type':'item', 'search':name, 'limit':limit,'headers':headers}
    return pywikibot.data.api.Request(site=site, parameters=params).submit()

def get_types(id, required_types=[], good_types=wd_whitelist_types, bad_types=wd_blacklist_types, endpoint=default_endpoint):
    """
    Given a wikidata id (e.g., Q7803487), returns a set of its types.
    """
    query = q_types_labels.format(QID=id)
    sparql = SPARQLWrapper(endpoint)
    sparql.setReturnFormat(JSON)
    sparql.setQuery(query)
    results = sparql.query().convert()
    types = set([])
    found_required_type = False if required_types else True
    for result in results["results"]["bindings"]:
        id_label = (wd_entity_id(result['type']['value']), result['typeLabel']['value'])
        id = id_label[0]
        if id in required_types:
            found_required_type = True
            types.add(id_label)
        if id in good_types:
            types.add(id_label)
        elif id in bad_types:
            return None
    if required_types and not found_required_type:
        return None
    return types



### Test examples
#prettyPrint(wd_search('acrobat'))
#prettyPrint(wd_search('ukraine'))
#prettyPrint(wd_search('anonymous'))
#prettyPrint(wd_search('ibm'))
#prettyPrint(wd_search('black hat'))
#prettyPrint(wd_search('Shadow Brokers'))
#prettyPrint(wd_search('julian assange'))
#prettyPrint(wd_search('ddos'))
#prettyPrint(wd_search('linux'))
#print(get_types('Q27134643'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('string', help='string to search for in a label or alias')
    p.add_argument('types', nargs='?', default='', help='comma seperate ids: "Q5,Q68"')
    args = p.parse_args()
    qtypes = [t for t in args.types.split(',')] if args.types else []
    prettyPrint(wd_search(args.string, required_types=qtypes))

