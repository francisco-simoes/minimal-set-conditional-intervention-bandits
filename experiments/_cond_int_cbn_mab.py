class CondIntCBN_MAB:
    """Encodes CBN-MAB problem for conditional interventions."""

    # TODO: This class needs to contain information about:
    #   1. the nodes that we can select from.
    #   2. context for each node (ancestors or something maybe more sophisticated later on).
    #   3. A reward CPD for each node, to be fed to a ContextualUCB per node.
    #
    #   NOTE: This seems to mean that the __init__ method should have the BN (from pgmpy)
    #   as an argument to extract the nodes and the ancestors of each node; OR: this class
    #   could subclass the BN class. The reward CPD should also be easy to extract from
    #   the CBN.
    pass
