import numpy as np

from numba import njit, f8, i8, b1
from numba.pycc import CC

try:
    from ..entities.soilProfile import SoilProfileNT_typ_sig
except:
    from entities.soilProfile import SoilProfileNT_typ_sig
    
# temporary name for compiled module
cc = CC("solution_rainfall_partition")


@cc.export("rainfall_partition", (f8,f8[:],i8,f8,f8,f8,f8,f8,f8,f8,f8,SoilProfileNT_typ_sig))
def rainfall_partition(
    precipitation,
    InitCond_th,
    NewCond_DaySubmerged,
    FieldMngt_SRinhb,
    FieldMngt_Bunds,
    FieldMngt_zBund,
    FieldMngt_CNadjPct,
    Soil_CN,
    Soil_AdjCN,
    Soil_zCN,
    Soil_nComp,
    prof,
):
    """
    Function to partition rainfall into surface runoff and infiltration using the curve number approach


    <a href="../pdfs/ac_ref_man_3.pdf#page=57" target="_blank">Reference Manual: rainfall partition calculations</a> (pg. 48-51)



    *Arguments:*


    `precipitation`: `float` : Percipitation on current day

    `InitCond`: `InitCondClass` : InitCond object containing model paramaters

    `FieldMngt`: `FieldMngtStruct` : field management params

    `Soil_CN`: `float` : curve number

    `Soil_AdjCN`: `float` : adjusted curve number

    `Soil_zCN`: `float` :

    `Soil_nComp`: `float` : number of compartments

    `prof`: `SoilProfileClass` : Soil object


    *Returns:*

    `Runoff`: `float` : Total Suface Runoff

    `Infl`: `float` : Total Infiltration

    `NewCond`: `InitCondClass` : InitCond object containing updated model paramaters






    """

    # can probs make this faster by doing a if precipitation=0 loop

    ## Store initial conditions for updating ##
    # NewCond = InitCond

    ## Calculate runoff ##
    if (FieldMngt_SRinhb == False) and ((FieldMngt_Bunds == False) or (FieldMngt_zBund < 0.001)):
        # Surface runoff is not inhibited and no soil bunds are on field
        # Reset submerged days
        NewCond_DaySubmerged = 0
        # Adjust curve number for field management practices
        CN = Soil_CN * (1 + (FieldMngt_CNadjPct / 100))
        if Soil_AdjCN == 1:  # Adjust CN for antecedent moisture
            # Calculate upper and lowe curve number bounds
            CNbot = round(
                1.4 * (np.exp(-14 * np.log(10)))
                + (0.507 * CN)
                - (0.00374 * CN ** 2)
                + (0.0000867 * CN ** 3)
            )
            CNtop = round(
                5.6 * (np.exp(-14 * np.log(10)))
                + (2.33 * CN)
                - (0.0209 * CN ** 2)
                + (0.000076 * CN ** 3)
            )
            # Check which compartment cover depth of top soil used to adjust
            # curve number
            comp_sto_array = prof.dzsum[prof.dzsum >= Soil_zCN]
            if comp_sto_array.shape[0] == 0:
                comp_sto = int(Soil_nComp)
            else:
                comp_sto = int(Soil_nComp - comp_sto_array.shape[0])

            # Calculate weighting factors by compartment
            xx = 0
            wrel = np.zeros(comp_sto)
            for ii in range(comp_sto):
                if prof.dzsum[ii] > Soil_zCN:
                    prof.dzsum[ii] = Soil_zCN

                wx = 1.016 * (1 - np.exp(-4.16 * (prof.dzsum[ii] / Soil_zCN)))
                wrel[ii] = wx - xx
                if wrel[ii] < 0:
                    wrel[ii] = 0
                elif wrel[ii] > 1:
                    wrel[ii] = 1

                xx = wx

            # Calculate relative wetness of top soil
            wet_top = 0
            # prof = prof

            for ii in range(comp_sto):
                th = max(prof.th_wp[ii], InitCond_th[ii])
                wet_top = wet_top + (
                    wrel[ii] * ((th - prof.th_wp[ii]) / (prof.th_fc[ii] - prof.th_wp[ii]))
                )

            # Calculate adjusted curve number
            if wet_top > 1:
                wet_top = 1
            elif wet_top < 0:
                wet_top = 0

            CN = round(CNbot + (CNtop - CNbot) * wet_top)

        # Partition rainfall into runoff and infiltration (mm)
        S = (25400 / CN) - 254
        term = precipitation - ((5 / 100) * S)
        if term <= 0:
            Runoff = 0
            Infl = precipitation
        else:
            Runoff = (term ** 2) / (precipitation + (1 - (5 / 100)) * S)
            Infl = precipitation - Runoff

    else:
        # bunds on field, therefore no surface runoff
        Runoff = 0
        Infl = precipitation

    return Runoff, Infl, NewCond_DaySubmerged

if __name__ == "__main__":
    cc.compile()
