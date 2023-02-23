def itype():
    map_ = {'Rill/Sprinkler/Wheel Line': 'Sprinkler',
            'Center Pivot/Drip': 'Pivot',
            'Drip/None': 'Drip',
            'Flood': 'Flood',
            'Rill/Wheel Line': 'Sprinkler',
            'Rill/Sprinkler': 'Sprinkler',
            'Drip/Rill': 'Drip',
            'Center Pivot/Rill/Wheel Line': 'Pivot',
            'Rill': 'Flood',
            'Big Gun/Drip': 'Sprinkler',
            'Center Pivot': 'Pivot',
            'Big Gun/Center Pivot': 'Sprinkler',
            'Drip/Rill/Sprinkler': 'Drip',
            'Drip/Sprinkler': 'Drip',
            'Wheel Line': 'Sprinkler',
            'Center Pivot/Rill': 'Pivot',
            'Micro-Sprinkler': 'Sprinkler',
            'Hand': 'Sprinkler',
            'Big Gun': 'Sprinkler',
            'Drip': 'Drip',
            'Big Gun/Wheel Line': 'Sprinkler',
            'Big Gun/Sprinkler': 'Sprinkler',
            'Drip/Big Gun': 'Drip',
            'Drip/Micro-Sprinkler': 'Drip',
            'Sprinkler/Wheel Line': 'Sprinkler',
            'Center Pivot/Sprinkler': 'Pivot',
            'Drip/Sprinkler/Wheel Line': 'Drip',
            'Center Pivot/None': 'Pivot',
            'Sprinkler': 'Sprinkler',
            'Center Pivot/Sprinkler/Wheel Line': 'Pivot',
            'Hand/Sprinkler': 'Sprinkler',
            'Center Pivot/Rill/Sprinkler': 'Pivot',
            'Drip/Wheel Line': 'Drip',
            'Center Pivot/Wheel Line': 'Pivot',
            'Micro Sprinkler': 'Sprinkler',
            'DRIP': 'Drip',
            'SPRINKLER': 'Sprinkler',
            'FURROW': 'Flood',
            'FLOOD': 'Flood',
            'P': 'Pivot',
            'D': 'Drip',
            'S': 'Sprinkler',
            'F': 'Flood',
            'GRATED_PIPE': 'None',
            'UNKNOWN': 'None',
            }
    return map_


def itype_integer_mapping():
    return {'None': 0,
            'Sprinkler': 1,
            'Pivot': 2,
            'Drip': 3,
            'Flood': 4}


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
