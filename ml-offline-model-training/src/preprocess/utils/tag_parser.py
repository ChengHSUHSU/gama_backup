# Align user profile generation tag mapping table
# Reference: https://gitlab.beango.com/datascience/user-profile-generation/-/blob/master/src/utils/utils.py#L462
ENTITY_TYPE_MAP = {
    'ner': {
        'PERSON': 'person',
        'EDITOR': 'person',
        'LOC': 'location',
        'LOCATION': 'location',
        'GPE': 'location',
        'EVENT': 'event',
        'ORG': 'organization',
        'PRODUCT': 'item',
        'NORP': 'others'
    }
}


def ner_parser(data: dict):
    """
    Args:
        data (dict): NER tag dictionary.
                    e.g. data = {'PERSON-伊能靜': 1.0, 'PERSON-小哈利': 5.0, 'PERSON-庾澄慶': 1.0, 'LOCATION-台北': 1.0}
    Returns:
        result (dict): list of flattened NER tag.
                    e.g. result = {'person': ['伊能靜', '小哈利', '庾澄慶'], 'location': ['台北'], 'event': [], 'organization': [], 'item': [], 'others': []}
    """

    # initialize mapping table
    result = {'person': [], 'location': [], 'event': [], 'organization': [], 'item': [], 'others': []}

    if data:
        for content in data:
            ner_tagging = content.split('-')[0]

            text_start_pos = content.find('-')
            ner_text = content[text_start_pos+1:]

            if ner_tagging in ENTITY_TYPE_MAP['ner']:
                ner_tagging = ENTITY_TYPE_MAP['ner'][ner_tagging]
            result[ner_tagging].append(ner_text)
    return result


def flatten_ner(x: dict):
    """
    Args:
        x (dict): NER tag dictionary.
                    e.g. x = {'person': ['伊能靜', '小哈利', '庾澄慶'], 'location': ['台北'], 'event': [], 'organization': [], 'item': [], 'others': []}
    Returns:
        result (list): list of flattened NER tag.
                    e.g. result = ['庾澄慶', '小哈利', '伊能靜', '台北']
    """
    entities = set()
    for row in x.values():
        entities = entities.union(row)
    return list(entities)
