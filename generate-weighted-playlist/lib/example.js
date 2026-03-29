const soundTypeRules =
    [
        {
            'musical_grouping': KICKSTAB,
            'dualpan_partners': [],
            'musical_patterns': {
                'volumes': [LOUD],
                'bpms': [FAST],
                'rhythm_patterns': [
                    [
                        {
                            'musical_duration': QUARTER_NOTE,
                            'possible_pannings': [HARD_LEFT],
                        }
                    ],
                    [
                        {
                            'musical_duration': QUARTER_NOTE,
                            'possible_pannings': [HARD_LEFT],
                        },
                        {
                            'musical_duration': QUARTER_NOTE,
                            'possible_pannings': [HARD_RIGHT],
                        },
                    ],
                    [
                        {
                            'musical_duration': QUARTER_NOTE,
                            'possible_pannings': [HARD_LEFT],
                        },
                        {
                            'musical_duration': QUARTER_NOTE_REST,
                            'possible_pannings': [HARD_CENTER],
                        },
                    ],
                ],
            },
        }
    ]