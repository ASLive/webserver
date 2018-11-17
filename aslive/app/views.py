# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from django.utils.safestring import mark_safe
import json

def index(request):
    return render(request, 'app/index.html', {})

def room(request, room_name):
    return render(request, 'app/room.html', {
        'room_name_json': mark_safe(json.dumps(room_name))
    })
