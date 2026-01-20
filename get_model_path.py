from hashlib import sha1
from fairseq2.assets import get_asset_store

cache_dir = '~/.cache/fairseq2/assets/'


# method from `fairseq2.assets.download_manager`
def _get_uri_hash(uri):
    h = sha1(str(uri).encode()).hexdigest()
    return h[:24]


def print_model_path(model_card_name):
    store = get_asset_store()
    card = store.retrieve_card(model_card_name)
    uri = card.metadata.get('checkpoint') or card.metadata.get('tokenizer')
    if not uri:
        print(f'Could not find a URI for {model_card_name}.')
        return
    folder_name = _get_uri_hash(uri)
    full_path = f'{cache_dir}{folder_name}/'
    print(f'Model card:   {model_card_name}')
    print(f'Download URL: {uri}')
    print(f'Local folder: {full_path}')
    print()


for model_card_name in [
    'omniASR_CTC_300M_v2',
    'omniASR_CTC_1B_v2',
    'omniASR_LLM_Unlimited_300M_v2',
    'omniASR_LLM_Unlimited_1B_v2',
    'omniASR_tokenizer_written_v2',
]:
    print_model_path(model_card_name)
