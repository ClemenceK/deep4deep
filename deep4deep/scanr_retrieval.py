import requests

# unused
# would be useful for a model focusing on French companies but
# not for a model applicable to international companies

# API refs here: https://stackoverflow.com/questions/65106756/using-the-scanr-api-from-gouv-fr-for-a-post-request-that-given-a-company-name/65115927#65115927

# For translation:
# from googletrans import Translator #https://pypi.org/project/googletrans/
# or
# from translate import Translator

def scanr_from_name_to_output(company_name, page_size=10):
    '''
    given a name, return a ScanR json output,
    ie a dict with keys: ['request', 'total', 'results', 'facets']
    '''
    url_structures = "https://scanr-api.enseignementsup-recherche.gouv.fr/api/v2/structures/search"

    params = {
       "pageSize": page_size,
       "query": company_name
    }
    scanr_output = requests.post(url_structures, json=params).json()
    return scanr_output

def scanr_from_output_to_id(scanr_output):
    '''
    from an output given by scanr_from_name_to_output, returns the company's scanr id
    '''
    return scanr_output['results'][0]['value']['id']


def scanr_from_id_to_en_description(company_id=833714694):
    '''

    # given a ScanR company id, returns the 'description' in English,
    or in French if no english is available,
    (translation lines are desactivated for now)
    '''

    base_url = "https://scanr-api.enseignementsup-recherche.gouv.fr/api/v2/structures/structure/"
    url = base_url + str(company_id)
    response = requests.get(url).json()

    english = response['description']['en']

    # if no english, send a request to a translation module


    if not english:
        print(f"for company {company_id}, could only retrieve French description. Sending to translation.")
        french = response['description']['fr']
        #======= for googletrans module
        # currently having a documented issue sending error:
        # https://stackoverflow.com/questions/52455774/googletrans-stopped-working-with-error-nonetype-object-has-no-attribute-group
        # AttributeError: 'NoneType' object has no attribute 'group'
        #translator = Translator()
        #english = translator.translate(french, dest='en', src='fr')

        #======= for translate module
        # https://pypi.org/project/translate/
        #translator = Translator(to_lang='en', from_lang='fr',) #with a limited number of calls a day
        #english = translator.translate(french)

        # if no API available: return the french text until a solution is found
        print(f"for company {company_id}, could not get translation. Keeping French.")
    englishenglish = french

    return english

def scanr_explore_output(scanr_output, n=3):
    '''
    exploration to help you see what's in the 'results' of scanr_output
    stops after showing the n first results
    '''
    print("What's in scanr_output['results']? A list containing a number of dict, all with the same 2 keys\n")

    for i, item in enumerate(scanr_output['results'][:n]):
        print(f"item {i} in results has keys: {item.keys()}")
        # results yields pageSize items:
            # of kind dict
            # with 2 keys: highlights (list) and value (dict)
            # each being a list
        print(f"length of list highlights: {len(item['highlights'])}")
        print(f"last element of highlights: {item['highlights'][len(item['highlights'])-1]}\n")

        print(f"what's in dict 'value': {item['value'].keys()}")
        print(f"in 'id' inside values: {item['value']['id']}")
        print("\n")

    return None





