import asyncio

# Step 1: Function a
async def a():
    await asyncio.sleep(1)  # simulate work
    print("a() finished")
    return "result_from_a"

# Step 2: Function b (depends on a)
async def b(x):
    await asyncio.sleep(2)  # simulate longer work
    print(f"b() received: {x}")
    return f"b_processed_{x}"

# Step 3: Function c (depends on a)
async def c(x):
    await asyncio.sleep(1)  # simulate shorter work
    print(f"c() received: {x}")
    return f"c_processed_{x}"

# Step 4: Main async workflow
async def main():
    # Run a() first
    result_a = await a()

    # Run b() and c() in parallel using result_a
    result_b, result_c = await asyncio.gather(
        b(result_a),
        c(result_a),
    )

    return result_a, result_b, result_c

# Run it
if __name__ == "__main__":
    results = asyncio.run(main())
    print("Final results:", results)
    print(type(results))
    print(results[0])
    print(results[1])
    print(results[2])
