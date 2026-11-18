import pynq
rails = pynq.get_rails()
print(list(rails.keys()))
for name, r in rails.items():
    print(f"{name:20s} V={r.voltage.value if r.voltage else '-':>8}  "
          f"I={r.current.value if r.current else '-':>8}  "
          f"P={r.power.value if r.power else '-':>8}")