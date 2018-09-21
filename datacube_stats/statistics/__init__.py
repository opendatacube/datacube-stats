"""
Classes for performing statistical data analysis.
"""

from .core import Statistic, PerPixelMetadata, SimpleStatistic
from .core import StatsConfigurationError, StatsProcessingError

from .incremental import MaskMultiCounter
from .external import ExternalPlugin
from .geomedian import GEOMEDIAN_STATS

from .uncategorized import ReducingXarrayStatistic, NoneStat
from .uncategorized import Percentile, PercentileNoProv
from .uncategorized import Medoid, MedoidNoProv, MedoidSimple
from .uncategorized import NormalisedDifferenceStats
from .uncategorized import WofsStats
from .uncategorized import TCWStats

from .mangrove import MangroveCC

try:
    from .geomedian import GeoMedian
except ImportError:
    pass


try:
    from .geomedian import NewGeomedianStatistic
except ImportError:
    pass


try:
    from .geomedian import SpectralMAD
except ImportError:
    pass


STATS = {
    'simple': ReducingXarrayStatistic,
    'percentile': Percentile,
    'percentile_no_prov': PercentileNoProv,
    'medoid': Medoid,
    'medoid_no_prov': MedoidNoProv,
    'medoid_simple': MedoidSimple,
    'simple_normalised_difference': NormalisedDifferenceStats,
    'none': NoneStat,
    'wofs_summary': WofsStats,
    'tcwbg_summary': TCWStats,
    'masked_multi_count': MaskMultiCounter,
    'mangrove_canopy_cover': MangroveCC,
    'external': ExternalPlugin,
    **GEOMEDIAN_STATS
}
