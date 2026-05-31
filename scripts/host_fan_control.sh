#!/bin/bash
# Manual / auto fan control via /sys/class/hwmon PWM channels.
#
# RUN ON THE PROXMOX HOST (root). Will not work inside a VM unless the
# motherboard sensor chip has been explicitly passed through.
#
# Usage:
#   sudo bash host_fan_control.sh status       # list discovered PWM channels
#   sudo bash host_fan_control.sh full         # all fans to 100%
#   sudo bash host_fan_control.sh auto         # restore BIOS / driver auto
#   sudo bash host_fan_control.sh set 180      # set a specific PWM value 0..255
#
# Notes:
#   * PWM enable values are driver-dependent:
#       0 = no control (full speed on most chips)
#       1 = manual / software-controlled
#       2 = automatic fan control
#     "auto" tries 2 first, falls back to 0.
#   * If `status` lists nothing, you probably need to load a sensor driver:
#       sudo sensors-detect --auto
#       sudo modprobe nct6775   # or it87, k10temp, coretemp etc.

set -u
ACTION="${1:-status}"
VALUE="${2:-}"

list_pwms() {
    find /sys/class/hwmon -maxdepth 2 -name "pwm[0-9]*" \
        ! -name "*_enable" ! -name "*_mode" ! -name "*_auto*" \
        ! -name "*_min" ! -name "*_max" ! -name "*_freq" 2>/dev/null
}

show_status() {
    local found=0
    for pwm in $(list_pwms); do
        found=1
        local dir name pwm_now enable_now fan_input
        dir=$(dirname "$pwm")
        name=$(cat "$dir/name" 2>/dev/null || echo "?")
        pwm_now=$(cat "$pwm" 2>/dev/null || echo "?")
        enable_now=$(cat "${pwm}_enable" 2>/dev/null || echo "?")
        # Find the matching fanN_input if it exists
        local fan_file="${pwm/pwm/fan}_input"
        if [ -f "$fan_file" ]; then
            fan_input=$(cat "$fan_file")
            printf "  %s [%s]  pwm=%3s/255  enable=%s  rpm=%s\n" \
                "$pwm" "$name" "$pwm_now" "$enable_now" "$fan_input"
        else
            printf "  %s [%s]  pwm=%3s/255  enable=%s\n" \
                "$pwm" "$name" "$pwm_now" "$enable_now"
        fi
    done
    if [ "$found" -eq 0 ]; then
        echo "  (no PWM channels found — load a sensor driver, see header)"
    fi
}

write_all() {
    local pwm_val="$1"
    local enable_val="$2"
    local count=0
    for pwm in $(list_pwms); do
        local enable_file="${pwm}_enable"
        if [ -w "$enable_file" ]; then
            echo "$enable_val" > "$enable_file" 2>/dev/null \
                || echo "  warn: could not write $enable_val to $enable_file"
        fi
        if [ -w "$pwm" ] && [ "$pwm_val" != "skip" ]; then
            echo "$pwm_val" > "$pwm" 2>/dev/null \
                || echo "  warn: could not write $pwm_val to $pwm"
        fi
        count=$((count + 1))
    done
    echo "  applied to $count channel(s)"
}

case "$ACTION" in
    status)
        echo "PWM channels:"
        show_status
        ;;
    full)
        echo "Setting all fans to 100% (enable=1, pwm=255):"
        write_all 255 1
        echo ""
        echo "After:"; show_status
        ;;
    auto)
        echo "Restoring automatic fan control (enable=2, fallback enable=0):"
        for pwm in $(list_pwms); do
            local enable_file="${pwm}_enable"
            if [ -w "$enable_file" ]; then
                echo 2 > "$enable_file" 2>/dev/null \
                    || echo 0 > "$enable_file" 2>/dev/null \
                    || echo "  warn: could not restore $enable_file"
            fi
        done
        echo ""
        echo "After:"; show_status
        ;;
    set)
        if [ -z "$VALUE" ] || ! [[ "$VALUE" =~ ^[0-9]+$ ]]; then
            echo "usage: $0 set <0..255>"; exit 2
        fi
        if [ "$VALUE" -gt 255 ]; then echo "max 255"; exit 2; fi
        echo "Setting all fans to PWM=$VALUE (enable=1):"
        write_all "$VALUE" 1
        echo ""
        echo "After:"; show_status
        ;;
    *)
        echo "usage: $0 {status|full|auto|set <0..255>}"
        exit 2
        ;;
esac
