class MaxFront:
    max_front = {
            "Crypto": {
                "BTC": 2,
                "ETH": 1,
                "MBT": 2,
                "MET": 1
            },
            "Currency": {
                "6A": 1,
                "6B": 1,
                "6C": 2,
                "6E": 2,
                "6J": 1,
                "6M": 1,
                "6N": 1,
                "6S": 1,
                "DX": 1
            },
            "Energy": {
                "BRN": 2,
                "CL": 5,
                "GAS": 6,
                "GWM": 8,
                "HO": 1,
                "NG": 10,
                "RB": 11,
                "WBS": 28
            },
            "Volatility": {
                "VX": 2
            },
            "Equity": {
                "EMD": 1,
                "ES": 1,
                "FCE": 1,
                "FDAX": 1,
                "FESX": 2,
                "FSMI": 1,
                "FTDX": 1,
                "GD": 1,
                "HSI": 1,
                "HTW": 1,
                "KOS": 1,
                "LFT": 1,
                "M2K": 1,
                "MES": 2,
                "MHI": 1,
                "MNQ": 2,
                "MYM": 1,
                "NIY": 1,
                "NKD": 1,
                "NQ": 1,
                "RTY": 1,
                "SCN": 1,
                "SNK": 1,
                "SSG": 1,
                "SXF": 1,
                "YAP": 1,
                "YM": 1
            },
            "Metal": {
                "GC": 6,
                "HG": 4,
                "PA": 1,
                "PL": 1,
                "SI": 5
            },
            "Fixed Income": {
                "CGB": 1,
                "FBTP": 1,
                "FGBL": 1,
                "FGBM": 1,
                "FGBS": 1,
                "FGBX": 1,
                "FOAT": 1,
                "LLG": 1,
                "SJB": 1,
                "TN": 1,
                "UB": 1,
                "YIB": 5,
                "YIR": 8,
                "YXT": 1,
                "YYT": 1,
                "ZB": 1,
                "ZF": 1,
                "ZN": 1,
                "ZQ": 3,
                "ZT": 1
            },
            "Rates": {
                "CRA": 2,
                "LEU": 16,
                "SO3": 10,
                "SR3": 8
            },
            "Grain": {
                "AFB": 1,
                "AWM": 1,
                "KE": 2,
                "LWB": 2,
                "MWE": 2,
                "ZC": 5,
                "ZL": 6,
                "ZM": 6,
                "ZO": 2,
                "ZR": 1,
                "ZS": 6,
                "ZW": 4
            },
            "Soft": {
                "CC": 3,
                "CT": 4,
                "DC": 5,
                "KC": 4,
                "LBR": 2,
                "LCC": 5,
                "LRC": 3,
                "LSU": 3,
                "OJ": 2,
                "RS": 2,
                "SB": 4
            },
            "Meat": {
                "GF": 4,
                "HE": 5,
                "LE": 4
            }
    }

    @classmethod
    def set_max_front(cls, symbol, sector, front):
        max_front_value = cls.max_front[sector][symbol]
        return min(front, max_front_value)

    _front_by_sector = {        # all backtests below 2010...today
        'Fixed Income': 3,      # 0..3 rapidly increasing performance, 4: no data for Eurex symbols
        'Equity': 3,            # 0..3 rapidly increasing performance, 4: no data for Eurex symbols (the same as above)
                                #       With long only, even better. Can we trade 3-rd month micro futures ?
        'Currency': 4,          # 0..4 increasing performance, 5: no dollar index values
        'Rates': 5,             # 0..2 bad, 3..8 gut, but may be just a chance to get gut trades, I leave it at the middle=5
        'Volatility': 1,        # I would skip volatility for now
        'Crypto': 1,            # not much difference
        'Energy': 2,            # 0..2 not much difference, 3 is worse
        'Metal': 2,             # difference 0...3 is small
        'Grain': 3,             # 3 is good (4 even a bit better), but can we trade 3rd contract?
        'Soft': 1,              # 0 and 1 are better
        'Meat': 2               # 1 best, but very few trades (12), may be we will remove this sector
    }

    @classmethod
    def front_by_sector(cls, sector: str) -> int:
        return cls._front_by_sector[sector]
