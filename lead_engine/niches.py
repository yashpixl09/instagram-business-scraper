"""Niche registry.

SearchAPI's `google_maps` engine returns `type` and `types` as human-readable display
labels ("Hair salon", "Coffee shop", "Historical landmark"). It does not expose Google's
`type_id`. SerpApi, hitting the same Google data, does expose it -- and it is simply the
display label slugified: "Cold storage facility" -> `cold_storage_facility`. So slugifying
the label reconstructs Google's own machine id; we are recovering an omitted field, not
inventing a heuristic.

Two consequences shape this module:

    queries          what we ask Google for          -> SearchAPI `q`
    include/exclude  what counts as a match          -> qualification

Google's taxonomy is flat, but its labels are compound nouns with the *head noun last*:
`sheet_metal_contractor` -> contractor, `kitchen_furniture_store` -> store. The only
hierarchy the taxonomy carries lives at the end of the slug, so a suffix rule is the exact
analogue of Geoapify's prefix rule at the other end of the string. That is what lets
`manufacturer` reject all of retail without enumerating two hundred store types.

Precedence, stated once:

    exact include > exact exclude > suffix exclude > suffix include > neutral

Exact-include-first is deliberate: it is the per-slug exception that carves one type out of
a broad suffix rule without disabling the rule.

`strict` + `qualification_terms` is unchanged from the prototype and is taxonomy
independent: it is the second gate that stops a broad or wrong Google category from
substituting one niche for another. `allow_name_only` relaxes the *first* gate (type
evidence) for the two niches where Google's taxonomy has no faithful type -- but it can
never bypass an exclusion, because a disqualifying type is fatal before either gate runs.

The type vocabulary below is authored from Google's published label set and from what these
queries return in an Indian metro. Phase 2 captures one real SearchAPI response per niche
and corrects it before the qualification logic is trusted.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .models import Lead

QUALIFIES, NEUTRAL, DISQUALIFIES = 1, 0, -1

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_EVIDENCE_SEPARATORS_RE = re.compile(r"[_\-.]+")


@dataclass(frozen=True)
class NicheProfile:
    id: str
    label: str
    queries: tuple[str, ...]
    include_types: frozenset[str] = frozenset()
    exclude_types: frozenset[str] = frozenset()
    include_suffixes: tuple[str, ...] = ()
    exclude_suffixes: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    qualification_terms: tuple[str, ...] = ()
    strict: bool = False
    allow_name_only: bool = False
    demand_base: int = 18
    budget_base: int = 12
    offer: str = ""
    scan_multiplier: int = 1


class UnsupportedNicheError(ValueError):
    def __init__(self, value: str) -> None:
        self.value = value
        super().__init__(f"Unsupported niche: {value}")


def slugify_type(value: object) -> str:
    """Reconstruct Google's `type_id` from its display label.

    Accents matter: Google returns both "Cafe" and "Café" depending on locale, and both
    must yield `cafe`.
    """
    decomposed = unicodedata.normalize("NFKD", str(value))
    ascii_only = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _SLUG_RE.sub("_", ascii_only.lower()).strip("_")


def _profile(
    niche_id: str,
    label: str,
    *,
    queries: tuple[str, ...],
    include_types: frozenset[str] = frozenset(),
    exclude_types: frozenset[str] = frozenset(),
    include_suffixes: tuple[str, ...] = (),
    exclude_suffixes: tuple[str, ...] = (),
    aliases: tuple[str, ...] = (),
    terms: tuple[str, ...] = (),
    strict: bool = False,
    allow_name_only: bool = False,
    demand: int,
    budget: int,
    offer: str,
    scan_multiplier: int = 1,
) -> NicheProfile:
    return NicheProfile(
        id=niche_id,
        label=label,
        queries=queries,
        include_types=include_types,
        exclude_types=exclude_types,
        include_suffixes=include_suffixes,
        exclude_suffixes=exclude_suffixes,
        aliases=aliases,
        qualification_terms=terms,
        strict=strict,
        allow_name_only=allow_name_only,
        demand_base=demand,
        budget_base=budget,
        offer=offer,
        scan_multiplier=scan_multiplier,
    )


NICHE_PROFILES: dict[str, NicheProfile] = {
    # ── Food and drink ────────────────────────────────────────────────
    "cafe": _profile(
        "cafe", "Cafe",
        aliases=("coffee shop",),
        queries=("cafe", "coffee shop", "coffee house", "espresso bar", "tea room"),
        include_types=frozenset({
            "cafe", "coffee_shop", "coffee_store", "espresso_bar", "coffee_stand",
            "coffee_roasters", "tea_house", "tea_room", "cafeteria", "brunch_restaurant",
        }),
        exclude_types=frozenset({
            # Only types that prove a *different* business. `restaurant`, `bar`,
            # `ice_cream_shop`, `meal_takeaway` and friends are dropped: a cafe that also
            # serves meals, alcohol, desserts or takeaway is the same shop, and Google
            # dual-labels almost every real cafe that way. A place typed only `restaurant`
            # is still rejected -- it carries no cafe include type.
            "night_club", "hotel",                      # nightlife and lodging venues
            "internet_cafe", "cyber_cafe",              # the browse/print shop, not a cafe
            "coffee_machine_supplier", "coffee_wholesaler",
        }),
        exclude_suffixes=("_supplier", "_wholesaler", "_distributor", "_manufacturer"),
        demand=20, budget=13,
        offer="online menu, table enquiries, and repeat-customer capture",
    ),
    "bakery": _profile(
        "bakery", "Bakery",
        queries=("bakery", "bakers", "bread shop", "artisan bakery", "iyengar bakery"),
        include_types=frozenset({
            "bakery", "bread_shop", "pastry_shop", "bagel_shop", "sourdough_bakery",
            "cookie_shop", "donut_shop",
        }),
        exclude_types=frozenset({
            # Iyengar bakeries are routinely typed {Bakery, Fast food restaurant} and the
            # neighbourhood bakery {Bakery, Grocery store} -- one business, two labels, so
            # those exclusions are gone. `supermarket` and `hotel` stay: the in-store
            # bakery counter belongs to the chain, and the pitch would go to the wrong firm.
            "hotel", "supermarket",
            "flour_mill",                               # upstream chakki, not a bakery
            "bakery_equipment_supplier", "food_products_supplier",
        }),
        exclude_suffixes=("_supplier", "_wholesaler", "_distributor", "_manufacturer",
                          "_equipment_store"),
        demand=23, budget=15,
        offer="product catalog, custom-order form, and advance payments",
    ),
    "cake_shop": _profile(
        "cake_shop", "Cake Shop",
        aliases=("cakes", "confectionery"),
        queries=("cake shop", "cake studio", "patisserie", "custom cakes", "dessert shop"),
        include_types=frozenset({
            "cake_shop", "cake_decorating_service", "dessert_shop", "patisserie",
            "chocolate_shop", "confectionery", "cupcake_shop", "candy_store",
            "chocolate_cafe",
        }),
        exclude_types=frozenset({
            # A cake studio that also delivers, or that Google also tags `wedding planner`
            # because it does wedding cakes, is still a cake studio -- those exclusions are
            # gone. The supply trade stays: it sells *to* this niche rather than being it.
            "hotel", "supermarket",
            "cake_decorating_equipment_shop", "baking_supply_store",
            "chocolate_factory", "confectionery_wholesaler",
        }),
        exclude_suffixes=("_supplier", "_wholesaler", "_distributor", "_manufacturer",
                          "_factory"),
        demand=23, budget=15,
        offer="occasion catalog, customization form, and advance payments",
    ),
    "cloud_kitchen": _profile(
        "cloud_kitchen", "Cloud Kitchen",
        aliases=("delivery kitchen", "ghost kitchen"),
        queries=("cloud kitchen", "delivery only restaurant", "ghost kitchen",
                 "takeaway kitchen", "food delivery kitchen"),
        include_types=frozenset({          # deliberately broad; terms do the narrowing
            "restaurant", "meal_delivery", "meal_takeaway", "delivery_restaurant",
            "takeout_restaurant", "food_delivery_service", "fast_food_restaurant",
            "cloud_kitchen",
            # `caterer` is deliberately absent: it is catering's defining type, and it was
            # the single type both niches claimed. Owning it here made every caterer look
            # like a cloud kitchen. A genuinely-named ghost kitchen that Google types only
            # as `caterer` still matches through `allow_name_only` on the terms below.
        }),
        exclude_types=frozenset({
            # THE anti-substitution case for this niche. Every entry here is load-bearing
            # because `allow_name_only` means *no* type evidence is required: without the
            # exclusion, a name carrying "cloud kitchen" would qualify a modular-kitchen
            # showroom outright. This is the profile where exclusions are the only defence.
            "kitchen_furniture_store", "modular_kitchen_store", "kitchen_supply_store",
            "kitchen_remodeler", "kitchen_planning_consultant", "furniture_store",
            "interior_designer", "cafe", "coffee_shop", "bar", "night_club",
            "bakery", "grocery_store", "supermarket", "hotel",
        }),
        exclude_suffixes=("_store", "_showroom", "_dealer", "_contractor"),
        terms=("cloud kitchen", "delivery kitchen", "ghost kitchen",
               "takeaway kitchen", "delivery only", "virtual kitchen"),
        strict=True, allow_name_only=True, demand=24, budget=14, scan_multiplier=3,
        offer="direct ordering, online payments, and WhatsApp confirmations",
    ),
    "catering": _profile(
        "catering", "Catering",
        aliases=("caterer", "catering service"),
        queries=("catering service", "caterers", "event catering", "banquet catering",
                 "tiffin service"),
        include_types=frozenset({
            "caterer", "catering_food_and_drink_supplier", "wedding_caterer",
            "corporate_catering_service", "box_lunch_supplier", "tiffin_center",
            "tiffin_service",
        }),
        include_suffixes=("_caterer",),
        exclude_types=frozenset({
            # The caterer-plus-banquet-hall operator and the restaurant with a catering arm
            # are the two highest-budget segments of this niche, and Google dual-labels
            # exactly those. `banquet_hall`, `wedding_venue`, `restaurant`, `event_planner`
            # and the rest of that family are therefore gone, along with the `_hall`/`_venue`
            # suffix rules that were re-imposing them. Precision is unharmed: a plain
            # restaurant carries no `caterer` type, so `type_ok` is False and a strict
            # profile without `allow_name_only` rejects it.
            "hotel",                                    # the hotel is the business, not the caterer
            "catering_equipment_supplier", "kitchen_supply_store",
        }),
        exclude_suffixes=("_store", "_showroom"),
        terms=("catering", "caterer", "banquet", "event food", "tiffin service"),
        strict=True, demand=23, budget=18,
        offer="menu catalog, event quotation requests, and tasting bookings",
    ),
    # ── Appointment-driven services ───────────────────────────────────
    "salon": _profile(
        "salon", "Salon",
        aliases=("hair salon", "beauty salon"),
        queries=("hair salon", "beauty salon", "unisex salon", "barber shop",
                 "nail studio", "beauty parlour"),
        include_types=frozenset({
            "hair_salon", "beauty_salon", "nail_salon", "barber_shop", "hairdresser",
            "beautician", "beauty_parlour", "beauty_parlor", "unisex_hairdresser",
            "waxing_hair_removal_service", "hair_removal_service", "make_up_artist",
            "eyebrow_bar", "hair_care", "threading_service",
        }),
        exclude_types=frozenset({
            "cafe", "coffee_shop", "restaurant", "bar", "bakery", "hotel",
            "gym", "fitness_center", "clothing_store",
            "beauty_supply_store", "cosmetics_store",      # retail, not service - different pitch
            "pet_groomer", "dog_grooming_service",         # "grooming" is in our terms
            # `beauty_school` is gone: the parlour that also trains is one business and the
            # bigger of the two. `dermatologist`/`skin_care_clinic`/`hair_transplant_clinic`
            # are gone too -- the medi-aesthetics place is genuinely both, and `clinic`
            # already refuses it from the other side, so it lands here rather than nowhere.
        }),
        terms=("salon", "hairdresser", "hair stylist", "barber", "beauty parlour",
               "beauty parlor", "nail studio", "unisex", "grooming"),
        strict=True, demand=21, budget=16, scan_multiplier=2,
        offer="service menu, appointment booking, and automated reminders",
    ),
    "spa": _profile(
        "spa", "Spa",
        aliases=("wellness spa",),
        queries=("spa", "massage spa", "wellness centre", "day spa", "ayurvedic massage"),
        include_types=frozenset({
            "spa", "day_spa", "massage_spa", "beauty_spa", "wellness_center",
            "massage_therapist", "foot_massage_parlor", "thai_massage_therapist",
            "reflexologist", "aromatherapy_service", "sauna", "steam_bath",
        }),
        exclude_types=frozenset({
            # "spa" returns hotels with spas, hospitals and equipment dealers.
            #
            # `ayurvedic_clinic` is deliberately NOT here. Google's own `_clinic` suffix
            # makes it qualify for `clinic`, which in turn refuses anything `_spa`; with
            # both exclusions in place the Kerala-style Ayurveda centre -- the highest-budget
            # place in this niche -- matched nothing at all. Spa yields to clinic on the
            # medical labels (`doctor`, `medical_clinic`, `physiotherapist`, `dermatologist`
            # are gone as well) and clinic yields to spa on the wellness ones, so every
            # dual-labelled place lands in exactly one of them.
            "hotel", "resort_hotel", "guest_house", "lodging",
            "gym", "fitness_center", "yoga_studio", "sports_club",
            "hospital", "cosmetic_surgeon", "plastic_surgeon",
            "beauty_supply_store", "spa_and_health_club_equipment_supplier",
            "massage_supply_store",
        }),
        exclude_suffixes=("_store", "_supplier", "_dealer", "_hotel", "_hospital"),
        terms=("spa", "massage", "sauna", "wellness", "ayurvedic treatment",
               "beauty treatment"),
        strict=True, demand=20, budget=18, scan_multiplier=2,
        offer="treatment catalog, appointment booking, and package enquiries",
    ),
    "fitness_gym": _profile(
        "fitness_gym", "Fitness/Gym",
        aliases=("fitness", "gym", "fitness gym"),
        queries=("gym", "fitness centre", "yoga studio", "crossfit box",
                 "personal training studio"),
        include_types=frozenset({
            "gym", "fitness_center", "fitness_centre", "physical_fitness_program",
            "yoga_studio", "pilates_studio", "personal_trainer", "sports_club",
            "boxing_gym", "crossfit_box", "aerobics_class", "zumba_class",
        }),
        exclude_types=frozenset({
            # "gym" returns equipment dealers and hotel fitness rooms; those stay out.
            # Everything a gym plausibly runs in-house -- spa, massage, physio, nutrition,
            # dance and martial-arts classes -- has been dropped. Each was neutral anyway
            # (a standalone spa or physio carries no gym type and is rejected regardless),
            # so keeping them only cost the dual-labelled {Gym, Spa} and {Gym, Dance school}
            # places, which are gyms.
            "gym_equipment_supplier", "sporting_goods_store", "sportswear_store",
            "hotel", "resort_hotel",
        }),
        exclude_suffixes=("_store", "_supplier", "_dealer", "_manufacturer", "_hotel"),
        terms=("gym", "fitness", "yoga", "pilates", "strength training", "fitness centre",
               "fitness center", "crossfit"),
        strict=True, demand=22, budget=16,
        offer="membership plans, trial bookings, and lead follow-up",
    ),
    "dental_clinic": _profile(
        "dental_clinic", "Dental Clinic",
        aliases=("dentist", "dental"),
        queries=("dental clinic", "dentist", "orthodontist", "dental implants",
                 "cosmetic dentistry"),
        include_types=frozenset({
            "dentist", "dental_clinic", "dental_hygienist", "orthodontist",
            "periodontist", "endodontist", "oral_surgeon", "pediatric_dentist",
            "cosmetic_dentist", "dental_implants_periodontist", "denture_care_center",
            "dental_radiology",
        }),
        include_suffixes=("_dentist",),
        exclude_types=frozenset({
            # "dentist" returns labs, suppliers and general hospitals.
            # `medical_clinic` and `doctor` are gone: {Dentist, Medical clinic} is how
            # Google labels a great many Indian dental practices, and `clinic` refuses that
            # place from its side, so keeping both exclusions orphaned it. A GP clinic with
            # no `dentist` type still fails `type_ok` here and is rejected.
            "dental_laboratory", "dental_supply_store", "dental_equipment_supplier",
            "hospital", "general_hospital", "pharmacy",
            "veterinarian", "veterinary_clinic", "medical_college",
        }),
        exclude_suffixes=("_store", "_supplier", "_laboratory", "_manufacturer",
                          "_hospital", "_college"),
        terms=("dental", "dentist", "orthodont", "smile", "tooth", "teeth"),
        strict=True, demand=22, budget=20,
        offer="treatment menu, appointment booking, and recall reminders",
    ),
    "clinic": _profile(
        "clinic", "Clinic",
        aliases=("doctor", "polyclinic", "medical clinic"),
        queries=("clinic", "polyclinic", "doctor", "general physician",
                 "physiotherapy clinic"),
        include_types=frozenset({
            "medical_clinic", "doctor", "general_practitioner", "family_practice_physician",
            "physiotherapist", "dermatologist", "pediatrician", "gynecologist",
            "walk_in_clinic", "polyclinic", "internist", "ophthalmologist",
            "eye_care_center", "skin_care_clinic",
        }),
        include_suffixes=("_clinic", "_physician", "_doctor", "_specialist"),
        exclude_types=frozenset({
            # Every adjacent niche that Google also labels "... clinic". This is the widest
            # exclusion set in the registry and it survives the audit intact for one reason:
            # `_clinic` is an *include suffix* here, so `dental_clinic`, `veterinary_clinic`,
            # `pet_clinic` and `hair_transplant_clinic` would all QUALIFY without the exact
            # exclusion, and the rest name a different profession or an institution. Every
            # dual-label these used to orphan is now caught by the counterpart niche
            # instead: {Dentist, Medical clinic} -> dental_clinic, {Beauty salon, Skin care
            # clinic} -> salon, {Massage spa, Ayurvedic clinic} -> spa, {Gym, Medical
            # clinic} -> fitness_gym. Clinic keeps the medical lane pure; the others took
            # the dual-labelled place.
            "dentist", "dental_clinic", "orthodontist", "cosmetic_dentist",
            "veterinarian", "veterinary_care", "veterinary_clinic", "animal_hospital",
            "pet_clinic", "pet_groomer",
            "spa", "day_spa", "massage_spa", "wellness_center", "massage_therapist",
            "gym", "fitness_center", "yoga_studio",
            "hair_salon", "beauty_salon", "hair_transplant_clinic",
            "hospital", "general_hospital", "pharmacy", "medical_supply_store",
            "diagnostic_laboratory", "medical_laboratory",
        }),
        exclude_suffixes=("_store", "_supplier", "_hospital", "_dentist", "_laboratory",
                          "_salon", "_spa"),
        terms=("clinic", "doctor", "physician", "medical", "physiotherapy", "dermatology",
               "pediatric", "healthcare"),
        strict=True, demand=21, budget=19,
        offer="doctor profiles, appointment booking, and follow-up reminders",
    ),
    "veterinary": _profile(
        "veterinary", "Veterinary",
        aliases=("vet", "pet clinic", "animal hospital"),
        queries=("veterinary clinic", "pet clinic", "animal hospital", "veterinary doctor",
                 "pet hospital"),
        include_types=frozenset({
            "veterinarian", "veterinary_care", "veterinary_clinic", "animal_hospital",
            "veterinary_hospital", "pet_clinic", "emergency_veterinarian_service",
            "veterinary_pharmacy", "pet_groomer",
        }),
        exclude_types=frozenset({
            # "pet clinic" returns pet retail and shelters far more than vets.
            # `pet_boarding_service` is gone -- the clinic-with-boarding is one business.
            "pet_store", "pet_supply_store", "pet_food_store", "aquarium_shop",
            "animal_shelter", "pet_adoption_service", "zoo",
            "doctor", "medical_clinic", "hospital", "dentist", "pharmacy",
        }),
        exclude_suffixes=("_store", "_supplier", "_dealer"),
        terms=("vet", "veterinary", "animal", "pet care", "pet clinic"),
        strict=True, demand=20, budget=17,
        offer="service list, appointment booking, and vaccination reminders",
    ),
    "auto_service": _profile(
        "auto_service", "Auto Service",
        aliases=("car service", "bike service", "garage"),
        queries=("car service centre", "bike service centre", "car repair",
                 "auto garage", "car wash"),
        include_types=frozenset({
            "auto_repair_shop", "car_repair_and_maintenance_service",
            "motorcycle_repair_shop", "auto_body_shop", "car_wash", "tire_shop",
            "wheel_alignment_service", "oil_change_service", "auto_electrical_service",
            "car_detailing_service", "brake_shop", "transmission_shop",
            "auto_air_conditioning_service", "auto_glass_shop",
        }),
        include_suffixes=("_repair_shop", "_repair_service", "_service_center"),
        exclude_types=frozenset({
            # "car service" returns dealers, parts shops, rentals and driving schools
            "car_dealer", "used_car_dealer", "motorcycle_dealer", "truck_dealer",
            "auto_parts_store", "car_accessories_store", "tire_manufacturer",
            "car_rental_agency", "taxi_service", "gas_station", "petrol_pump",
            "auto_insurance_agency", "driving_school",
            "auto_auction", "car_leasing_service",
            # `towing_service` is gone: the garage that also tows is the same garage.
        }),
        exclude_suffixes=("_dealer", "_store", "_showroom", "_school", "_rental_agency",
                          "_manufacturer", "_wholesaler"),
        terms=("service centre", "service center", "auto", "car", "bike", "motor", "garage",
               "repair", "workshop"),
        strict=True, demand=20, budget=17,
        offer="service menu, slot booking, and service-due reminders",
    ),
    "preschool": _profile(
        "preschool", "Preschool",
        aliases=("play school", "playschool", "daycare", "day care"),
        queries=("preschool", "play school", "day care", "montessori school",
                 "creche"),
        include_types=frozenset({
            "preschool", "play_school", "day_care_center", "kindergarten",
            "nursery_school", "child_care_agency", "montessori_school", "creche",
            "early_childhood_education_center",
        }),
        exclude_types=frozenset({
            # "play school" returns toy retail and leisure venues; those stay out.
            #
            # The K-12 labels are gone, and this was the single largest recall loss in the
            # registry: Indian preschools are overwhelmingly typed {Preschool, Primary
            # school} or {Kindergarten, Private school} because the pre-primary wing sits
            # inside the school. Every one of those was orphaned. Nothing is lost -- this
            # profile has no include suffixes, so a school with no preschool type was
            # already failing `type_ok`. The same argument retires the tuition and
            # enrichment labels: `tutor_class` refuses anything preschool-shaped from its
            # side, so {Preschool, Tutoring service} now lands here rather than nowhere.
            "driving_school",
            "toy_store", "baby_store", "childrens_clothing_store",
            "pediatrician", "amusement_center",
        }),
        exclude_suffixes=("_store", "_college", "_university"),
        terms=("preschool", "play school", "playschool", "kindergarten", "day care",
               "daycare", "montessori", "creche"),
        strict=True, demand=21, budget=18,
        offer="programme details, admission enquiries, and campus-visit booking",
    ),
    "tutor_class": _profile(
        "tutor_class", "Tutor/Class",
        aliases=("tuition", "coaching class", "tutor"),
        queries=("tuition centre", "coaching classes", "academy", "training institute",
                 "music classes"),
        include_types=frozenset({
            "tutoring_service", "coaching_center", "educational_institution",
            "training_centre", "training_center", "music_school", "dance_school",
            "language_school", "art_school", "computer_training_school",
            "test_preparation_center", "learning_center", "tutor",
        }),
        include_suffixes=("_school", "_institute", "_academy", "_tutoring_service",
                          "_coaching_center", "_training_center"),
        exclude_types=frozenset({
            # Every other education-shaped niche the "_school" suffix would otherwise sweep
            # in -- fourteen of these are caught by that suffix and so would QUALIFY without
            # the exact exclusion. `drivers_license_training_school` is the third driving
            # variant Google emits; naming only the first two let it through the `_school`
            # suffix with "training" satisfying the strict gate, and a place typed solely
            # "Drivers license training school" was being returned for tutor requests.
            "driving_school", "motor_driving_school", "drivers_license_training_school",
            "preschool", "play_school", "kindergarten",
            "nursery_school", "montessori_school", "day_care_center", "child_care_agency",
            "creche", "beauty_school", "martial_arts_school", "swimming_school",
            "primary_school", "elementary_school", "high_school", "secondary_school",
            "private_school", "international_school", "college", "university",
            "medical_college", "engineering_college", "business_school",
            "gym", "fitness_center", "yoga_studio",
        }),
        exclude_suffixes=("_store", "_college", "_university", "_hostel"),
        terms=("tutor", "tuition", "coaching", "academy", "classes", "training", "language",
               "music", "dance", "test prep", "institute"),
        strict=True, demand=21, budget=13, scan_multiplier=2,
        offer="course catalog, demo-class booking, and student enquiries",
    ),
    "driving_school": _profile(
        "driving_school", "Driving School",
        aliases=("motor training", "driving classes"),
        queries=("driving school", "motor driving school", "car driving classes",
                 "driving instructor"),
        include_types=frozenset({
            "driving_school", "motor_driving_school", "driving_test_center",
            "driving_test_centre", "driving_instructor", "drivers_license_training_school",
        }),
        exclude_types=frozenset({
            # "driving school" returns RTO offices, garages and rentals.
            # The coaching labels are gone: Google types plenty of motor schools
            # {Driving school, Training centre}, and `tutor_class` excludes all three
            # driving variants from its side, so that place had nowhere to land.
            "auto_repair_shop", "car_repair_and_maintenance_service", "car_dealer",
            "used_car_dealer", "car_rental_agency", "taxi_service",
            "preschool", "transportation_service", "government_office",
            "vehicle_registration_office", "auto_insurance_agency",
        }),
        exclude_suffixes=("_dealer", "_store", "_showroom", "_repair_shop"),
        terms=("driving", "motor training", "learners"),
        strict=True, demand=19, budget=14,
        offer="course packages, slot booking, and enrolment enquiries",
    ),
    # ── Project and enquiry-driven services ───────────────────────────
    "photographer": _profile(
        "photographer", "Photographer",
        aliases=("photography", "photo studio"),
        queries=("photographer", "photography studio", "wedding photographer",
                 "portrait studio", "product photography"),
        include_types=frozenset({
            "photographer", "photography_studio", "photography_service",
            "wedding_photographer", "portrait_studio", "commercial_photographer",
            "product_photographer", "event_photographer", "videographer",
            "photo_studio",
        }),
        include_suffixes=("_photographer", "_photography_studio", "_photography_service"),
        exclude_types=frozenset({
            # Our "studio" term is broad, so every other kind of studio is named out.
            # Dropped: `passport_photo_service`, `photo_lab` and `photo_printing_service`
            # (the neighbourhood studio does all three -- one business), and the planner
            # labels (the wedding photographer tagged {Photographer, Wedding planner} was
            # orphaned, since `event_planner` refuses photographers from its side).
            "camera_store", "print_shop",
            "recording_studio", "dance_studio", "yoga_studio",
            "art_gallery", "art_studio", "tattoo_studio", "nail_salon",
            "camera_repair_shop", "video_equipment_rental_service",
        }),
        exclude_suffixes=("_store", "_supplier", "_dealer", "_rental_service"),
        terms=("photo", "photograph", "studio", "films", "captures", "lens", "shoot"),
        strict=True, demand=24, budget=16,
        offer="portfolio gallery, package pricing, and shoot enquiries",
    ),
    "event_planner": _profile(
        "event_planner", "Event Planner",
        aliases=("event management", "wedding planner"),
        queries=("event management company", "wedding planner", "event organiser",
                 "birthday party planner", "corporate event management"),
        include_types=frozenset({
            "event_planner", "wedding_planner", "party_planner",
            "event_management_company", "wedding_service", "corporate_event_planner",
            "event_organizer", "event_decorator",
        }),
        exclude_types=frozenset({
            # "event management" returns venues and rental firms -- those stay out, because
            # a hall is a venue business and this niche sells planning.
            #
            # The caterer, photographer and florist labels are gone. They were pure
            # dead weight: none is caught by an include rule here, so a caterer or
            # photographer with no planner type was already failing `type_ok`. All they did
            # was orphan "XYZ Events & Catering" and the planner who shoots her own
            # weddings -- one business wearing two of Google's labels.
            "banquet_hall", "function_room_facility", "wedding_venue", "event_venue",
            "convention_center", "hotel", "resort_hotel",
            "party_equipment_rental_service", "tent_rental_service", "party_store",
            "interior_designer", "furniture_store", "travel_agency",
        }),
        exclude_suffixes=("_store", "_hall", "_venue", "_rental_service", "_hotel"),
        terms=("event", "wedding", "planner", "decor", "celebration", "occasion"),
        strict=True, demand=23, budget=19,
        offer="portfolio, package enquiries, and quotation requests",
    ),
    "interior_designer": _profile(
        "interior_designer", "Interior Designer",
        aliases=("interior design", "interior decorator"),
        queries=("interior designer", "interior design firm", "turnkey interiors",
                 "home interior designer", "office interior designer"),
        include_types=frozenset({
            "interior_designer", "interior_decorator", "interior_architect_office",
            "interior_design_studio", "interior_fit_out_contractor",
            "office_interior_designer", "home_designer",
        }),
        exclude_types=frozenset({
            # "interior designer" returns furniture retail, modular-kitchen showrooms and
            # fashion designers - our "design" and "decor" terms would happily match all three
            "furniture_store", "home_goods_store", "modular_kitchen_store",
            "kitchen_furniture_store", "curtain_store", "lighting_store", "carpet_store",
            "furniture_maker", "furniture_manufacturer",
            "fashion_designer", "graphic_designer", "web_designer", "advertising_agency",
            "construction_company", "civil_engineer",
            # Gone: `general_contractor` (a turnkey interiors firm *is* one -- this profile
            # even includes `interior_fit_out_contractor`), the real-estate labels, and the
            # event labels. All three were neutral here and only cost dual-labelled places.
        }),
        exclude_suffixes=("_store", "_showroom", "_dealer", "_manufacturer", "_wholesaler"),
        terms=("interior", "design", "decor", "turnkey", "modular"),
        strict=True, demand=22, budget=21,
        offer="project portfolio, consultation booking, and quotation requests",
    ),
    "real_estate": _profile(
        "real_estate", "Real Estate",
        aliases=("property dealer", "realtor", "property consultant"),
        queries=("real estate agent", "property dealer", "real estate agency",
                 "property consultant", "flats for sale"),
        include_types=frozenset({
            "real_estate_agency", "real_estate_agent", "real_estate_consultant",
            "property_management_company", "real_estate_developer", "estate_agent",
            "commercial_real_estate_agency", "apartment_rental_agency",
            "real_estate_rental_agency",
        }),
        include_suffixes=("_real_estate_agency", "_estate_agent"),
        exclude_types=frozenset({
            # "property" returns the buildings themselves plus builders, banks and lawyers
            "apartment_complex", "apartment_building", "housing_society", "condominium_complex",
            "hotel", "resort_hotel", "guest_house", "hostel", "serviced_apartment",
            # `construction_company` and `home_builder` are gone: the Indian developer is
            # literally "XYZ Builders & Developers" and Google labels it both ways.
            "general_contractor", "civil_engineer",
            "interior_designer", "architect",
            "bank", "mortgage_lender", "loan_agency", "insurance_agency",
            "lawyer", "law_firm", "notary_public", "moving_company", "packers_and_movers",
        }),
        exclude_suffixes=("_store", "_contractor", "_hotel", "_bank"),
        terms=("real estate", "property", "properties", "realty", "estates", "builders",
               "homes"),
        strict=True, demand=21, budget=20,
        offer="listing pages, enquiry capture, and site-visit booking",
    ),
    "travel_agency": _profile(
        "travel_agency", "Travel Agency",
        aliases=("tour operator", "travel agent"),
        queries=("travel agency", "tour operator", "holiday packages",
                 "tour packages", "visa consultant"),
        include_types=frozenset({
            "travel_agency", "tour_operator", "tour_agency", "travel_agent",
            "travel_services", "holiday_package_provider", "visa_consultant",
            "tour_guide_service",
        }),
        exclude_types=frozenset({
            # THE substitution case here: Google answers "tour" with the places you tour
            "tourist_attraction", "historical_landmark", "monument", "museum",
            "national_park", "amusement_park", "zoo", "temple", "scenic_spot",
            "tourist_information_center",
            "hotel", "resort_hotel", "guest_house", "hostel", "lodging",
            "airline", "airport", "bus_station",
            # Rail ticketing and cab hire are what an Indian travel agency sells, so
            # `train_ticket_agency`, `car_rental_agency`, `taxi_service` and
            # `transportation_service` are gone -- a standalone cab operator carries no
            # travel-agency type and is still rejected.
            "immigration_and_naturalization_service", "passport_office",
        }),
        exclude_suffixes=("_hotel", "_landmark", "_attraction", "_museum", "_park",
                          "_station", "_store"),
        terms=("travel", "tour", "holiday", "trip", "voyage", "tourism", "getaway"),
        strict=True, demand=22, budget=17,
        offer="package catalog, itinerary enquiries, and booking requests",
    ),
    "professional_services": _profile(
        "professional_services", "Professional Services",
        aliases=("ca firm", "law firm", "consultant", "accountant"),
        queries=("chartered accountant", "law firm", "business consultant",
                 "tax consultant", "company registration consultant"),
        include_types=frozenset({
            "chartered_accountant", "certified_public_accountant", "accountant",
            "accounting_firm", "auditor", "tax_consultant", "tax_preparation_service",
            "lawyer", "law_firm", "legal_services", "notary_public",
            "company_secretary", "business_management_consultant",
            "management_consultant", "business_consultant", "financial_consultant",
        }),
        include_suffixes=("_lawyer", "_attorney", "_accountant"),
        exclude_types=frozenset({
            # "consultant" is the most over-applied label in Google's Indian data
            "real_estate_consultant", "real_estate_agency", "property_management_company",
            "visa_consultant", "immigration_consultant", "education_consultant",
            "interior_designer", "kitchen_planning_consultant",
            "insurance_agency", "bank", "loan_agency", "stock_broker",
            "software_company", "it_consultant", "marketing_agency",
            "advertising_agency", "employment_agency", "recruiter",
        }),
        exclude_suffixes=("_store", "_agency", "_school", "_manufacturer", "_designer"),
        terms=("associates", "consultant", "consultancy", "advisors", "advisory", "chartered",
               "accountant", "legal", "law", "tax", "audit", "company secretary"),
        strict=True, demand=19, budget=22,
        offer="service pages, consultation booking, and document intake",
    ),
    # ── Retail and production ─────────────────────────────────────────
    "boutique": _profile(
        "boutique", "Boutique",
        aliases=("fashion boutique",),
        queries=("boutique", "designer boutique", "womens clothing store",
                 "saree shop", "ethnic wear store"),
        include_types=frozenset({
            "boutique", "clothing_store", "womens_clothing_store", "mens_clothing_store",
            "childrens_clothing_store", "fashion_designer", "dress_store",
            "sari_store", "saree_shop", "bridal_shop", "tailor",
            "fashion_accessories_store",
        }),
        include_suffixes=("_clothing_store", "_wear_store", "_boutique"),
        exclude_types=frozenset({
            # retail chains and the upstream garment trade both answer "clothing store"
            "shopping_mall", "department_store", "supermarket", "hypermarket",
            "second_hand_store",
            "garment_exporter", "textile_mill", "clothing_manufacturer",
            "clothing_wholesaler", "laundry", "dry_cleaner",
            "interior_designer",
            # Gone: `fabric_store` (every saree shop sells cloth), `shoe_store`,
            # `sportswear_store`, `uniform_store` (adjacent apparel, one shop), and
            # `home_goods_store`/`furniture_store` (the lifestyle boutique is both, and
            # `home_decor` refuses clothing from its side).
        }),
        exclude_suffixes=("_manufacturer", "_wholesaler", "_exporter", "_mill", "_factory",
                          "_distributor", "_mall"),
        terms=("boutique", "fashion", "apparel", "clothes", "clothing", "garment", "couture",
               "designs", "accessories"),
        strict=True, demand=20, budget=16,
        offer="mobile catalog, product enquiries, and WhatsApp checkout",
    ),
    "home_decor": _profile(
        "home_decor", "Home Decor",
        aliases=("home decoration", "interior decor", "furniture"),
        queries=("home decor store", "furniture store", "home furnishing",
                 "modular kitchen showroom", "curtain shop"),
        include_types=frozenset({
            "home_goods_store", "furniture_store", "furniture_maker", "curtain_store",
            "lighting_store", "carpet_store", "rug_store", "mattress_store",
            "modular_kitchen_store", "kitchen_furniture_store", "home_furnishing_store",
            "handicraft_store",
        }),
        include_suffixes=("_furniture_store", "_furnishing_store", "_decor_store"),
        exclude_types=frozenset({
            # "furniture" collides with the factories that make it -- that half stands.
            # The interior and event labels are gone: the modular-kitchen showroom that
            # also designs, and the decor store that dresses weddings, are single
            # businesses, and `interior_designer` already excludes every `_store` from its
            # side, so those places were matching nothing at all.
            "furniture_manufacturer", "furniture_wholesaler", "furniture_repair_shop",
            "shopping_mall", "department_store", "hardware_store",
            "building_materials_store", "clothing_store", "general_contractor",
        }),
        exclude_suffixes=("_manufacturer", "_factory", "_wholesaler", "_contractor",
                          "_mall", "_mill"),
        terms=("home decor", "decor", "interior", "furniture", "furnishing", "lighting",
               "curtain", "carpet", "furnishings"),
        strict=True, demand=19, budget=17,
        offer="visual catalog, project enquiries, and quotation requests",
    ),
    "manufacturer": _profile(
        "manufacturer", "Manufacturer",
        aliases=("factory", "manufacturing"),
        queries=("manufacturer", "manufacturing company", "factory",
                 "fabrication works", "industrial supplier", "machine shop"),
        include_types=frozenset({
            "manufacturer", "machine_shop", "cnc_machine_shop", "metal_fabricator",
            "steel_fabricator", "sheet_metal_contractor", "metal_workshop", "foundry",
            "die_casting_company", "tool_and_die_shop", "plastic_products_supplier",
            "rubber_products_supplier", "packaging_company", "corrugated_box_supplier",
            "textile_mill", "garment_exporter", "chemical_manufacturer",
            "furniture_manufacturer", "food_products_supplier",
            "industrial_equipment_supplier",
        }),
        include_suffixes=("_manufacturer", "_factory", "_mill", "_fabricator",
                          "_foundry", "_industry", "_works"),
        exclude_types=frozenset({
            # As with `cloud_kitchen`, `allow_name_only` makes every entry here the only
            # type-based defence: "Sri Balaji Industries Pvt Ltd" is perfect name evidence,
            # so without the exclusion a warehouse or a logistics firm would qualify.
            "wholesaler", "distributor", "shopping_mall", "warehouse",
            "logistics_service", "freight_forwarding_service", "trucking_company",
            "courier_service", "software_company", "employment_agency",
        }),
        exclude_suffixes=("_store", "_dealer", "_wholesaler", "_distributor",
                          "_showroom", "_agency", "_restaurant"),
        terms=("manufacturer", "manufacturing", "factory", "works", "workshop",
               "production", "fabrication", "industries", "industrial", "mills",
               "engineering", "pvt ltd", "private limited"),
        strict=True, allow_name_only=True, demand=18, budget=20, scan_multiplier=3,
        offer="capability catalog, quotation requests, and qualified B2B enquiries",
    ),
}


def normalize_niche(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def resolve_niche_ids(values: list[str]) -> list[str]:
    lookup: dict[str, str] = {}
    for profile in NICHE_PROFILES.values():
        for alias in (profile.id, profile.label, *profile.aliases):
            lookup[normalize_niche(alias)] = profile.id

    resolved: list[str] = []
    for value in values:
        niche_id = lookup.get(normalize_niche(value))
        if not niche_id:
            raise UnsupportedNicheError(value)
        if niche_id not in resolved:
            resolved.append(niche_id)
    return resolved


def niche_payload() -> list[dict]:
    return [
        {
            "id": profile.id,
            "label": profile.label,
            "queries": list(profile.queries),
            "include_types": sorted(profile.include_types),
            "exclude_types": sorted(profile.exclude_types),
            "include_suffixes": list(profile.include_suffixes),
            "exclude_suffixes": list(profile.exclude_suffixes),
        }
        for profile in NICHE_PROFILES.values()
    ]


def classify_type(profile: NicheProfile, slug: str) -> int:
    """Verdict for a single Google type slug.

    Precedence: exact include > exact exclude > suffix exclude > suffix include > neutral.
    Exact include first is the per-slug exception that carves one type out of a broad
    suffix rule without disabling the rule.
    """
    if slug in profile.include_types:
        return QUALIFIES
    if slug in profile.exclude_types:
        return DISQUALIFIES
    if any(slug.endswith(suffix) for suffix in profile.exclude_suffixes):
        return DISQUALIFIES
    if any(slug.endswith(suffix) for suffix in profile.include_suffixes):
        return QUALIFIES
    return NEUTRAL


def has_name_evidence(profile: NicheProfile, lead: Lead) -> bool:
    evidence = " ".join([lead.name, lead.category, *lead.raw_categories]).lower()
    evidence = _EVIDENCE_SEPARATORS_RE.sub(" ", evidence)
    return any(term in evidence for term in profile.qualification_terms)


def matches_niche(profile: NicheProfile, lead: Lead) -> bool:
    """Two gates: the place's types must belong to the niche, and for strict profiles the
    place's own name or categories must carry corroborating evidence.

    One disqualifying type anywhere is fatal, before either gate. That is what stops a
    request for salons returning cafes, and it is why `allow_name_only` cannot be used to
    smuggle an excluded place back in.
    """
    slugs = [
        s
        for s in (
            slugify_type(lead.category),
            *(slugify_type(t) for t in lead.raw_categories),
        )
        if s
    ]
    verdicts = [classify_type(profile, s) for s in slugs]
    if DISQUALIFIES in verdicts:
        return False
    type_ok = QUALIFIES in verdicts
    if not profile.strict:
        return type_ok
    if not type_ok and not profile.allow_name_only:
        return False
    return has_name_evidence(profile, lead)
