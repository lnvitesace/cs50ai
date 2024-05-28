import csv
import itertools
import sys
from functools import reduce

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    probabilities = {}
    has_no_parents = {}
    has_parents = {}
    for name in people:
        if people[name]["mother"] is None:
            has_no_parents[name] = people[name]
        else:
            has_parents[name] = people[name]
    genes = get_gene_numbers(people, one_gene, two_genes)

    for name in has_no_parents:
        gene = genes[name]
        prob = PROBS["gene"][gene] * PROBS["trait"][gene][name in have_trait]
        probabilities[name] = prob

    for name in has_parents:
        mother = people[name]["mother"]
        father = people[name]["father"]
        mother_gene = genes[mother]
        father_gene = genes[father]
        gene = genes[name]

        if gene == 0:
            prob = parent_giving(mother_gene, False) * parent_giving(father_gene, False)
        elif gene == 1:
            prob = (parent_giving(mother_gene, False) * parent_giving(father_gene, True) +
                    parent_giving(mother_gene, True) * parent_giving(father_gene, False))
        else:
            prob = parent_giving(mother_gene, True) * parent_giving(father_gene, True)

        prob *= PROBS["trait"][gene][name in have_trait]
        probabilities[name] = prob

    return reduce(lambda x, y: x * y, probabilities.values())


def get_gene_numbers(people, one_gene, two_genes):
    """
    Compute the dictionary of the expect gene numbers for each person.
    """
    genes = {}
    for name in people:
        if name in one_gene:
            genes[name] = 1
        elif name in two_genes:
            genes[name] = 2
        else:
            genes[name] = 0
    return genes


def parent_giving(parent_gene, giving):
    """
    Compute the probability of a parent giving a gene to their child.
    """
    if giving:
        if parent_gene == 0:
            return PROBS["mutation"]
        elif parent_gene == 1:
            return 0.5
        else:
            return 1 - PROBS["mutation"]
    else:
        if parent_gene == 0:
            return 1 - PROBS["mutation"]
        elif parent_gene == 1:
            return 0.5
        else:
            return PROBS["mutation"]


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    genes = get_gene_numbers(probabilities, one_gene, two_genes)
    for name in probabilities:
        gene = genes[name]
        probabilities[name]["gene"][gene] += p
        probabilities[name]["trait"][name in have_trait] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for name in probabilities:
        total_gene = sum(probabilities[name]["gene"].values())
        total_trait = sum(probabilities[name]["trait"].values())
        for gene in probabilities[name]["gene"]:
            probabilities[name]["gene"][gene] /= total_gene
        for trait in probabilities[name]["trait"]:
            probabilities[name]["trait"][trait] /= total_trait


if __name__ == "__main__":
    main()
