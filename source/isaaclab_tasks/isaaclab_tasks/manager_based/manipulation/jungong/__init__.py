# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom (Jungong) manipulation tasks."""

# Import subpackages to trigger gym registrations.
from . import get_place_bandage  # noqa: F401
from . import clean_mortarbarrel 
from . import plant_redflag 
from . import insert_magazine
from . import sorting_bullets