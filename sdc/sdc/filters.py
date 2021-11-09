"""
** Filtering Info **

To filter scenes by tags one should specify a filter function
Scene tags dict has following structure:
{
    'day_time': one of {'kNight', 'kMorning', 'kAfternoon', 'kEvening'}
    'season': one of {'kWinter', 'kSpring', 'kSummer', 'kAutumn'}
    'track': one of {
      'Moscow' , 'Skolkovo', 'Innopolis', 'AnnArbor', 'Modiin', 'TelAviv'}
    'sun_phase': one of {'kAstronomicalNight', 'kTwilight', 'kDaylight'}
    'precipitation': one of {'kNoPrecipitation', 'kRain', 'kSleet', 'kSnow'}
}
Full description of protobuf message is available at
tags.proto file in sources

** Split Configuration **

Training Data ('train')
'moscow__train': Moscow intersected with NO precipitation

Development Data ('development')
'moscow__development': Moscow intersected with NO precipitation
'ood__development': Skolkovo, Modiin, and Innopolis intersected with
    (No precipitation, Rain and Snow)

Test Data ('test')
'moscow__test': Moscow intersected with NO precipitation
'ood__test': Ann-Arbor + Tel Aviv intersected with
    (No precipitation, rain, snow and sleet)
"""


def filter_moscow_no_precipitation_data(scene_tags_dict):
    """
    This will need to be further divided into train/validation/test splits.
    """
    if (scene_tags_dict['track'] == 'Moscow' and
            scene_tags_dict['precipitation'] == 'kNoPrecipitation'):
        return True
    else:
        return False


def filter_ood_development_data(scene_tags_dict):
    if (scene_tags_dict['track'] in ['Skolkovo', 'Modiin', 'Innopolis'] and
        scene_tags_dict[
            'precipitation'] in ['kNoPrecipitation', 'kRain', 'kSnow']):
        return True
    else:
        return False


DATASETS_TO_FILTERS = {
    'train': {
        'moscow__train': filter_moscow_no_precipitation_data
    },
    'development': {
        'moscow__development': filter_moscow_no_precipitation_data,
        'ood__development': filter_ood_development_data
    }
}
