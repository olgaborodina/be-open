#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from astroquery.vizier import Vizier

VizOC = Vizier(catalog="J/A+A/659/A59/table2")
VizOC.ROW_LIMIT = -1

OC_df = VizOC.query_constraints(cluster="Blanco_1").values()[0].to_pandas()
