import createClient from "openapi-fetch";
import type { paths } from "./schema.js";

export type RagtestClientOptions = Parameters<typeof createClient<paths>>[0];
export type RagtestClient = ReturnType<typeof createClient<paths>>;

export function createRagtestClient(
  options: RagtestClientOptions,
): RagtestClient {
  return createClient<paths>(options);
}
